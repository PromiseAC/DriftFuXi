"""Stage 1 only: actual official model/loss with an opt-in CPU ragged adapter."""
import argparse
import csv
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/fuxi-linear'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cpu-reference', action='store_true', help='Explicitly enable CPU-only ragged reference operators')
    p.add_argument('--debug-shapes', action='store_true', help='Print shapes on first training batch only')
    p.add_argument('--chunk-size', type=int, default=4, help='0 selects official non-chunk path')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/stage1/smoke.json')
    args = p.parse_args()
    if not args.cpu_reference:
        p.error('This local harness requires --cpu-reference; it does not validate CUDA/DDP')
    from cpu_jagged import install, check
    install()
    adapter_checks = check()
    from generative_recommenders.data.dataset import DatasetV2
    from generative_recommenders.trainer.data_loader import create_data_loader
    from generative_recommenders.modeling.sequential.features import movielens_seq_features_from_row
    from generative_recommenders.modeling.sequential.fuxi_linear import FuXiLinear
    from generative_recommenders.modeling.sequential.embedding_modules import LocalEmbeddingModule
    from generative_recommenders.modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
    from generative_recommenders.modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor
    from generative_recommenders.modeling.similarity.dot_product import DotProductSimilarity
    from generative_recommenders.modeling.sequential.autoregressive_losses import LocalNegativesSampler, SampledSoftmaxLoss
    from generative_recommenders.data.eval import get_eval_state, eval_metrics_v2_from_tensors
    from generative_recommenders.indexing.mips_top_k import MIPSBruteForceTopK

    torch.manual_seed(2026)
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture = ROOT/'artifacts/stage1/synthetic_sequences.csv'
    with fixture.open('w') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['user_id','sequence_item_ids','sequence_ratings','sequence_timestamps'])
        for u, length in enumerate([6,9,7,10,8,11]):
            # Positive IDs, unique within user; irregular, monotonic timestamps in ms.
            w.writerow([u, ','.join(str(1+(u*13+i)%64) for i in range(length)), ','.join(['4.0']*length), ','.join(str(1700000000000+u*10000+i*i*137) for i in range(length))])
    train = DatasetV2(str(fixture), padding_length=9, ignore_last_n=1, chronological=True)
    evaluation = DatasetV2(str(fixture), padding_length=9, ignore_last_n=0, chronological=True)
    _, loader = create_data_loader(train, 2, 1, 0, shuffle=False)
    _, eval_loader = create_data_loader(evaluation, 2, 1, 0, shuffle=False)
    D, I = 16, 64
    model = FuXiLinear(
        max_sequence_len=8, max_output_len=4, embedding_dim=D, num_blocks=2,
        num_heads=2, linear_dim=8, attention_dim=8, normalization='rel_bias',
        linear_activation='silu', linear_dropout_rate=0., attn_dropout_rate=0., ffn_multiply=1,
        embedding_module=LocalEmbeddingModule(I,D), similarity_module=DotProductSimilarity(),
        input_features_preproc_module=LearnablePositionalEmbeddingInputFeaturesPreprocessor(12,D,0.),
        output_postproc_module=L2NormEmbeddingPostprocessor(D),
        channel_t_config={'num_heads':2,'base':2,'base_stride':3,'start_index':10,'aug_current':True,'learnable_gamma':True},
        channel_p_config={'dim':4,'aug_current':True}, chunk_size=args.chunk_size or None, verbose=False)
    sampler = LocalNegativesSampler(I, model._embedding_module._item_emb, list(range(1,I+1)), True, 1e-6)
    loss_fn = SampledSoftmaxLoss(8, .05, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9,.98), weight_decay=0.)
    shapes, handles, gradients, losses, norms, updates = {}, [], {}, [], [], []
    first_batch = True
    logits_parts = []

    def record(name, tensor):
        if first_batch:
            shapes[name] = list(tensor.shape)
            if args.debug_shapes:
                print(f'SHAPE {name}: {list(tensor.shape)} {tensor.dtype}')
        assert torch.isfinite(tensor).all(), f'Non-finite {name}'

    def hook(name):
        def fn(module, inp, kwargs, out):
            record(name, out[0] if isinstance(out, tuple) else out)
            if name.endswith('ffn'):
                record(name+'.combined_gated_input', inp[0])
            if name.endswith('semantic'):
                for key in ('q', 'k', 'v'): record(name+'.'+key, kwargs[key])
            if name.endswith('temporal') and isinstance(out[1], dict):
                for key in ('query', 'key', 'log_decay_pos'):
                    if key in out[1]: record(name+'.'+key, out[1][key])
            if name.endswith('block'):
                record(name+'.input', kwargs['x'])
        return fn

    for name, mod in model.named_modules():
        tag = None
        if name.endswith('_retention'): tag = name+'.semantic'
        elif name.endswith('_channel_t'): tag = name+'.temporal'
        elif name.endswith('_channel_p'): tag = name+'.positional'
        elif name.endswith('_mffn'): tag = name+'.ffn'
        elif name in ['_fuxi._attention_layers.0','_fuxi._attention_layers.1']: tag = name+'.block'
        if tag: handles.append(mod.register_forward_hook(hook(tag), with_kwargs=True))
    def score_hook(module, inp, kwargs, out):
        logits_parts.append(out.detach())
        record('positive_logits' if out.shape[1] == 1 else 'negative_logits',out)
    handles.append(model._ndp_module.register_forward_hook(score_hook, with_kwargs=True))
    model.train()
    for step, row in enumerate(loader):
        features, targets, _ = movielens_seq_features_from_row(row, 'cpu', 4)
        features.past_ids.scatter_(1, features.past_lengths[:,None], targets)
        record('item_ids',features.past_ids)
        record('timestamps',features.past_payloads['timestamps'])
        record('sequence_lengths',features.past_lengths)
        optimizer.zero_grad(set_to_none=True)
        emb = model.get_item_embeddings(features.past_ids)
        record('embeddings',emb)
        hidden = model(features.past_lengths,features.past_ids,emb,features.past_payloads)
        record('hidden_states',hidden)
        labels = features.past_ids[:,1:]
        record('labels',labels)
        logits_parts.clear()
        loss = loss_fn(features.past_lengths,hidden[:,:-1],labels,emb[:,1:],(labels!=0).float(),sampler)
        record('loss',loss)
        # The official loss concatenates these after temperature scaling and collision masking.
        if first_batch:
            shapes['sampled_softmax_logits'] = [logits_parts[0].shape[0],9]
            if args.debug_shapes: print('SHAPE sampled_softmax_logits:',shapes['sampled_softmax_logits'])
        assert 0 <= loss.item() < 1e4, 'Smoke loss guard exceeded'
        loss.backward()
        sq = 0.
        for name, param in model.named_parameters():
            norm = None if param.grad is None else param.grad.norm().item()
            if first_batch: gradients[name] = {'grad_is_none':param.grad is None,'norm':norm}
            if norm is not None:
                assert torch.isfinite(param.grad).all(), name
                sq += norm**2
        total_norm = sq**.5
        assert total_norm < 1e4, 'Smoke gradient guard exceeded (not a convergence criterion)'
        groups = ['_channel_t','_channel_p','_retention','_uvqk','_mffn','_embedding_module']
        for group in groups:
            assert any(group in n and p.grad is not None and p.grad.norm()>0 for n,p in model.named_parameters()),group
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        changed = [n for n,p in model.named_parameters() if not torch.equal(p.detach(),before[n])]
        for group in groups: assert any(group in n for n in changed),group
        updates.append(len(changed)); norms.append(total_norm); losses.append(loss.item())
        print(f'STEP {step}: loss={loss.item():.8f}, gradient_norm={total_norm:.8f}, updated_parameters={len(changed)}')
        first_batch = False
    for handle in handles: handle.remove()
    model.eval()
    evaluated = 0
    metric_keys = set()
    with torch.inference_mode():
        state = get_eval_state(model,list(range(1,I+1)),sampler,MIPSBruteForceTopK,'cpu')
        for row in eval_loader:
            features, targets, ratings = movielens_seq_features_from_row(row,'cpu',4)
            # Official metric path, explicitly unfiltered because upstream's fixed k=I
            # cannot return I unseen items from a tiny catalog. Separately check filtered top-5.
            metrics = eval_metrics_v2_from_tensors(state,model,features,targets,target_ratings=ratings,filter_invalid_ids=False)
            assert all(torch.isfinite(v).all() for v in metrics.values())
            metric_keys.update(metrics)
            user = model.encode(features.past_lengths,features.past_ids,model.get_item_embeddings(features.past_ids),features.past_payloads)
            ids,scores,_ = state.candidate_index.get_top_k_outputs(user,5,state.top_k_module,features.past_ids)
            assert ids.shape == (2,5) and torch.isfinite(scores).all()
            assert not (ids[:,:,None] == features.past_ids[:,None,:]).any()
            shapes['final_user_representation'] = list(user.shape)
            shapes['candidate_embeddings'] = list(state.candidate_index.embeddings.shape)
            shapes['filtered_top_k'] = list(ids.shape)
            evaluated += len(targets)
    result = {'status':'PASS_CPU_REFERENCE_SYNTHETIC_ONLY','seed':2026,'chunk_size':args.chunk_size or None,
              'steps':len(losses),'losses':losses,'gradient_norms':norms,'updated_parameter_tensors':updates,
              'gradients_first_batch':gradients,'shapes':shapes,'adapter_checks':adapter_checks,
              'evaluation':{'users':evaluated,'official_metrics_filter_invalid_ids':False,'finite_metric_keys':sorted(metric_keys),'separate_seen_filtered_top_k':5},
              'numeric_checks':{'finite_forward_loss_grad_parameters':True,'loss_guard':10000,'gradient_norm_guard':10000},
              'limitations':['synthetic data, not dataset reproduction','CPU reference ragged ops, not native FBGEMM','No CUDA, NCCL, performance or convergence validation']}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],args.output)

if __name__ == '__main__':
    main()

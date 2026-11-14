## 2026-11-14 08:45:17 UTC: `/kaggle/working/brink/experiments/pipeline.py train --config configs/pipeline.yaml --out results/pipeline --work results/_work/pipeline`

```
Traceback (most recent call last):
  File "/kaggle/working/brink/brink/io.py", line 97, in record_failure
    yield
  File "/kaggle/working/brink/experiments/pipeline.py", line 388, in main
    train_stage(cfg, Path(args.out), work or Path(args.out).parent / "_work" / Path(args.out).name)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/brink/experiments/pipeline.py", line 257, in train_stage
    dataset = curate(cfg, data, fp, loader, raw, cdir, work) if cond in GATED else raw
              ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/brink/experiments/pipeline.py", line 86, in curate
    ).run()
      ~~~^^
  File "/usr/local/lib/python3.13/dist-packages/curatorkit/pipeline.py", line 152, in run
    passed, rejected = step.run(samples)
                       ~~~~~~~~^^^^^^^^^
  File "/kaggle/working/brink/brink/gate/curatorkit_gate.py", line 66, in run
    scores = self._score([sample_to_pair(s) for s in samples])
  File "/kaggle/working/brink/brink/gate/curatorkit_gate.py", line 54, in _score
    return np.asarray(self.scorer.score(pairs)[self.scorer.primary])
                      ~~~~~~~~~~~~~~~~~^^^^^^^
  File "/kaggle/working/brink/brink/gate/scorer.py", line 297, in score
    return grad_scores(self.model, self.tok, pairs, self.g_targets, self.max_len, log_every, on_progress)
  File "/kaggle/working/brink/brink/gate/scorer.py", line 194, in grad_scores
    out = model(**batch)
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/peft/peft_model.py", line 2120, in forward
    return self.base_model(
           ~~~~~~~~~~~~~~~^
        input_ids=input_ids,
        ^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/peft/tuners/tuners_utils.py", line 359, in forward
    return self.model(*args, **kwargs)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/utils/generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/models/cohere2/modeling_cohere2.py", line 477, in forward
    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/loss/loss_utils.py", line 72, in ForCausalLMLoss
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/loss/loss_utils.py", line 42, in fixed_cross_entropy
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/functional.py", line 3504, in cross_entropy
    return torch._C._nn.cross_entropy_loss(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        input,
        ^^^^^^
    ...<5 lines>...
        label_smoothing,
        ^^^^^^^^^^^^^^^^
    )
    ^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 870.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 228.81 MiB is free. Including non-PyTorch memory, this process has 14.34 GiB memory in use. Of the allocated memory 14.13 GiB is allocated by PyTorch, and 70.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```

## 2026-11-14 08:45:21 UTC: `experiments/pipeline.py --config configs/pipeline.yaml --out results/pipeline`

```
Traceback (most recent call last):
  File "/kaggle/working/brink/brink/io.py", line 97, in record_failure
    yield
  File "/kaggle/working/brink/experiments/pipeline.py", line 390, in main
    run(cfg, Path(args.out), work, config_path=Path(args.config))
    ~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/brink/experiments/pipeline.py", line 352, in run
    subprocess.run(
    ~~~~~~~~~~~~~~^
        [
        ^
    ...<11 lines>...
        check=True,
        ^^^^^^^^^^^
    )
    ^
  File "/usr/lib/python3.13/subprocess.py", line 578, in run
    raise CalledProcessError(retcode, process.args,
                             output=stdout, stderr=stderr)
subprocess.CalledProcessError: Command '['/usr/bin/python3', '/kaggle/working/brink/experiments/pipeline.py', 'train', '--config', 'configs/pipeline.yaml', '--out', 'results/pipeline', '--work', 'results/_work/pipeline']' returned non-zero exit status 1.
```


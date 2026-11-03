"""The serving engine: scheduler, batching policies, prefix cache and CUDA Graph decode.

Start with `chunked_prefill.ChunkedPrefillPagedRunner`, the current scheduler. It inherits
`continuous_prefix.ContinuousPrefixPagedRunner`, which inherits `prefix_cache.PrefixCachedPagedRunner`.
"""

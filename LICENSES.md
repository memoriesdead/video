# License scope

Licenses apply per component. The top-level license does not replace upstream terms.

| Files | Terms |
| --- | --- |
| Original top-level `tools/` and `tests/` code and original documentation | Apache License 2.0, in [LICENSE](LICENSE) |
| `backends/wangp/` patch, adapter and derived tests | [WanGP Community License 2.0](backends/wangp/LICENSE.txt); [MMGP notice](backends/wangp/mmgp-LICENSE.md) |

Modified WanGP code is identified in the backend README, modification notice and
patch, dated 2026-10-30. WanGP terms include conditions on redistribution and paid
services. The Apache license for the top-level tools does not grant Apache rights
to the WanGP-derived component.

Separately installed runtime dependencies and model weights retain their own
licenses. No model weights, CUDA runtime binaries, FFmpeg binaries or vendored
runtime dependency trees are distributed in this source package.

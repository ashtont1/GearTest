

Sometimes, when you call model.generate, you will get the following error:
"RuntimeError: The size of tensor a (4096) must match the size of tensor b (6473) at non-singleton dimension 3."

This most likely is a version mismatch.

For example, when I run run-fast.sh in GEAR environment (gear-pip-list.text), I see this error.
However, the problem goes away when I raun in cachegen environment (cachegenenv-pip-list.txt)



# Install and uninstall

Tensward needs Python 3.11 or newer (`requires-python = ">=3.11"`) and Linux or macOS. The
package depends on `httpx`, `packaging` and `pydantic`. The optional `gpu` extra adds
`nvidia-ml-py` for GPU queries on Linux.

## Recommended: uv

```sh
uv tool install tensward
tensward --version
```

uv fetches a suitable Python itself when the system one is older. It installs from
`curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pipx install uv`); open a new shell
afterwards so `uv` and the tools it installs are on your `PATH`.

## pipx

```sh
pipx install tensward
tensward --version
```

pipx comes from your package manager (`sudo apt install pipx && pipx ensurepath`).

## Virtualenv

```sh
python3.11 -m venv ~/tw-venv && . ~/tw-venv/bin/activate
pip install tensward
```

## Ubuntu 22.04

Ubuntu 22.04 ships Python 3.10, so `pip install tensward` finds no matching version. Use
`uv tool install tensward`, which downloads a Python 3.11 or newer for the tool, or install a
newer Python and create the virtualenv with it.

## Debian and Ubuntu: "externally managed environment"

On a current Debian or Ubuntu, `pip install tensward` outside a virtualenv fails with
`error: externally managed environment` (PEP 668: the system Python belongs to the package
manager). Do not force it with `--break-system-packages`; use uv, pipx or a virtualenv as above.

## Offline install

On a machine with network access and the same Python version and platform as the target, build
a wheelhouse of the wheel and everything it depends on:

```sh
pip download tensward --dest wheelhouse
# add "tensward[gpu]" to also fetch nvidia-ml-py
```

Copy `wheelhouse/` to the target and install from it only:

```sh
python3.11 -m venv ~/tw-venv && . ~/tw-venv/bin/activate
pip install --no-index --find-links wheelhouse tensward
```

To install a wheel you built yourself, put it in the same directory (`uv build` writes it to
`dist/`) and run the same command. Tensward does not download models or pull images, so the
engine image and the model must also be on the target. The engine itself is not part of the
wheelhouse.

## From source

```sh
git clone https://github.com/Tensward/tensward && cd tensward
python3.11 -m venv .venv && . .venv/bin/activate
pip install .
```

## Uninstall

```sh
uv tool uninstall tensward        # or: pipx uninstall tensward
rm -rf ~/.cache/tensward ./.tensward
```

`~/.cache/tensward` is Tensward's cache (`$XDG_CACHE_HOME/tensward` when that is set) and
`./.tensward` is the project folder in the current directory, with its runs and reports. A
project registered with `--project <dir>` keeps its data in `<dir>`; remove that folder too.
Knowledge records you created locally are in `~/.tensward/kb` (or `TENSWARD_KB_HOME`).

With the Docker runtime, the engine's compile caches under
`~/.cache/tensward/engine-cache/<image id>/` are written by the container as root, so the
`rm -rf` above cannot delete them. Remove them with
`sudo rm -rf ~/.cache/tensward/engine-cache`, then run the `rm -rf` line.

In a virtualenv, `pip uninstall tensward` removes the package.

# Fallback for non-flake setups: `nix-shell`
#
# Dependencies come from nixpkgs rather than pip. On NixOS a pip-installed manylinux wheel
# with a compiled extension links against paths that do not exist, so `make venv` is the
# wrong entry point there.
{ pkgs ? import <nixpkgs> { } }:

let
  # See flake.nix: fastapi's test-suite dependency has broken builds on some nixpkgs
  # revisions, and we are not developing fastapi.
  python = pkgs.python312.withPackages (ps: with ps; [
    (fastapi.overridePythonAttrs (_: { doCheck = false; }))
    uvicorn
    httpx
    pydantic
    pyyaml
    rich
    pytest
  ]);
in
pkgs.mkShell {
  packages = [ python pkgs.ruff pkgs.gnumake ];

  shellHook = ''
    echo "AgentTrace dev shell - python $(python --version | cut -d' ' -f2)"
    echo "  make test | make report | make serve"
    unset PYTHONPATH
  '';
}

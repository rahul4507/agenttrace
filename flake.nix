{
  description = "AgentTrace - coverage and regression analysis for voice agents";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          # Dependencies come from nixpkgs rather than pip. On NixOS a pip-installed
          # manylinux wheel with a compiled extension (pydantic-core, ruff) links against
          # paths that do not exist, so `make venv` is the wrong entry point there.
          # fastapi's own test suite pulls inline-snapshot, which has itself failed to
          # build on some nixpkgs revisions and takes the whole closure down with it.
          # We are not developing fastapi, so its checks are not ours to run.
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
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.ruff      # standalone binary, not the pip wheel
              pkgs.gnumake   # so the documented `make` targets work
            ];

            shellHook = ''
              echo "AgentTrace dev shell - python $(python --version | cut -d' ' -f2)"
              echo ""
              echo "  make test      81 tests, no network"
              echo "  make report    coverage report"
              echo "  make serve     dashboard on http://127.0.0.1:8124"
              echo ""
              echo "Skip make if you prefer: python -m pytest,"
              echo "python -m agenttrace.cli report --offline"
              echo ""
              # The Makefile falls back to `python3` from PATH when no .venv exists,
              # which is what we want here.
              unset PYTHONPATH
            '';
          };
        });

      # `nix run` starts the dashboard without entering a shell.
      apps = forAllSystems (pkgs:
        let
          python = pkgs.python312.withPackages (ps: with ps; [
            (fastapi.overridePythonAttrs (_: { doCheck = false; }))
            uvicorn httpx pydantic pyyaml rich
          ]);
        in
        {
          default = {
            type = "app";
            program = "${pkgs.writeShellScript "agenttrace-serve" ''
              exec ${python}/bin/python -m uvicorn agenttrace.api:app \
                --host 127.0.0.1 --port ''${PORT:-8124}
            ''}";
          };
        });
    };
}

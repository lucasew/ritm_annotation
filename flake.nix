{
  description = "Tool to do dataset annotation for semantic segmentation datsets";

  inputs = {
    nixpkgs.url = "nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
  flake-utils.lib.eachDefaultSystem (system: let
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in {
      overlay = import ./nix/overlay.nix;
      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          gnumake
          python3Packages.black
          python3Packages.mypy
          python3Packages.flake8
          python3Packages.pytest
          python3Packages.pytorch-bin
          python3Packages.opencv4
        ];
      };
    });
}

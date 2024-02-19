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
      shellHook = ''
        PYTHONPATH="$PYTHONPATH:$(pwd)"
      '';
      packages = {
        default = pkgs.python3Packages.callPackage ./package.nix { };
        default-cuda = pkgs.python3Packages.callPackage ./package.nix {
          pytorch = pkgs.python3Packages.pytorch-bin;
          torchvision = pkgs.python3Packages.torchvision-bin;
        };
      };
      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          gnumake
          gettext # i18n
          # dev
          python3Packages.pylsp-mypy
          python3Packages.isort
          python3Packages.black
          python3Packages.mypy
          python3Packages.flake8
          python3Packages.pytest
          # runtime
          python3Packages.cython
          python3Packages.easydict
          python3Packages.albumentations
          python3Packages.pillow
          python3Packages.scipy
          python3Packages.tensorboard
          python3Packages.tkinter
          python3Packages.tqdm
          python3Packages.pytorch-bin
          python3Packages.torchvision-bin
          python3Packages.opencv4
          python3Packages.pycocotools
        ];
      };
    });
}

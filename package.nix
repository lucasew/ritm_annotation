{ buildPythonPackage
, cython
, easydict
, albumentations
, pillow
, scipy
, tensorboard
, hatchling
, tkinter
, pytorch-bin
, torchvision-bin
, opencv4
, tqdm
, pycocotools
, pytestCheckHook
, fetchFromGitHub
}:

let
  albumentations_1 = albumentations.overrideDerivation (old: rec {
    version = "1.4.24";
 
    src = fetchFromGitHub {
      owner = "albumentations-team";
      repo = "albumentations";
      tag = version;
      hash = "sha256-2bZSuVECfJiAJRwVd0G93bjDdWlyVOpqf3AazQXTiJw=";
    };

    patches = [];

    postPatch = (old.postPatch or "") + ''
      printf 'def check_for_updates() -> None:\n\tpass' >> albumentations/check_version.py
    '';
  });
in

buildPythonPackage {
  pname = "ritm_annotation";
  version = builtins.readFile ./ritm_annotation/VERSION;
  pyproject = true;
  build-system = [ hatchling ];

  src = ./.;

  nativeBuildInputs = [ cython ];

  propagatedBuildInputs = [
    easydict
    albumentations_1
    pillow
    scipy
    tensorboard
    tkinter
    pytorch-bin
    pycocotools
    torchvision-bin
    opencv4
    tqdm
  ];

  checkInputs = [ pytestCheckHook ];

  pythonImportsCheck = [ "ritm_annotation" ];
}

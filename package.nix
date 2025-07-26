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
  albumentations_1 = albumentations.overrideDerivation rec {
    version = "1.4.24";
 
    src = fetchFromGitHub {
      owner = "albumentations-team";
      repo = "albumentations";
      tag = version;
      # hash = "sha256-8vUipdkIelRtKwMw63oUBDN/GUI0gegMGQaqDyXAOTQ=";
    };
  };
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

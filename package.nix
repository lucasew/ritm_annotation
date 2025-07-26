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
}:

buildPythonPackage {
  pname = "ritm_annotation";
  version = builtins.readFile ./ritm_annotation/VERSION;
  pyproject = true;
  build-system = [ hatchling ];

  src = ./.;

  nativeBuildInputs = [ cython ];

  propagatedBuildInputs = [
    easydict
    albumentations
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

{ buildPythonPackage
, wrapPython
, pytest
, cython
, easydict
, albumentations
, pillow
, scipy
, tensorboard
, tkinter
, pytorch
, torchvision
, opencv4
}:

buildPythonPackage {
  pname = "ritm-annotation";
  version = builtins.readFile ./ritm_annotation/VERSION;
  src = ./.;

  propagatedBuildInputs = [
    easydict
    albumentations
    pillow
    scipy
    tensorboard
    tkinter
    pytorch
    torchvision
    opencv4
  ];

  nativeBuildInputs = [ cython ];

  checkInputs = [ pytest ];

  pythonImportsCheck = [ "ritm_annotation" ];
}

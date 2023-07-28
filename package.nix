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
, pythonRelaxDepsHook
}:

buildPythonPackage {
  pname = "ritm-annotation";
  version = builtins.readFile ./ritm_annotation/VERSION;
  src = ./.;

  nativeBuildInputs = [ cython ];

  postPatch = ''
    substituteInPlace requirements.txt \
      --replace 'opencv-python-headless' "" \
      --replace 'torchvision >= 0.15.2' ""
  '';

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

  checkInputs = [ pytest ];

  pythonImportsCheck = [ "ritm_annotation" ];
}

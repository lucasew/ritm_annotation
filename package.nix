{ buildPythonPackage
, wrapPython
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
, pytestCheckHook
}:

buildPythonPackage {
  pname = "ritm_annotation";
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

  checkInputs = [ pytestCheckHook ];

  pythonImportsCheck = [ "ritm_annotation" ];
}

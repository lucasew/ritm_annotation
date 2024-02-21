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
, tqdm
, pycocotools
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
    pycocotools
    torchvision
    opencv4
    tqdm
  ];

  checkInputs = [ pytestCheckHook ];

  pythonImportsCheck = [ "ritm_annotation" ];
}

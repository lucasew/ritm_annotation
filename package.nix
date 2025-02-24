{ buildPythonPackage
, cython
, easydict
, albumentations
, pillow
, scipy
, tensorboard
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
    pytorch-bin
    pycocotools
    torchvision-bin
    opencv4
    tqdm
  ];

  checkInputs = [ pytestCheckHook ];

  pythonImportsCheck = [ "ritm_annotation" ];
}

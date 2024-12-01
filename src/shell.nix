{ pkgs ? import <nixpkgs> {} }:
let
  pythonEnv = pkgs.python312.withPackages (ps: with ps; [
    tabula-py
    pandas
    pdf2image
    pytesseract
    pillow
    opencv4
    numpy
    scipy
    pip
    matplotlib
  ]);
in pkgs.mkShell {
  buildInputs = with pkgs; [
    pythonEnv
    jre8
    poppler_utils  # This includes pdftoppm and other utilities
    tesseract  # For OCR functionality
  stdenv.cc.cc.lib  # This should provide libstdc++.so.6
  ];
  shellHook = ''
    export JAVA_HOME=${pkgs.jre8}
    export LD_LIBRARY_PATH=${pkgs.jre8}/lib/openjdk/lib/server:$LD_LIBRARY_PATH

    export PATH=${pkgs.poppler_utils}/bin:$PATH

    echo "Checking critical dependencies..."
    echo "pdftoppm version: $(pdftoppm -v 2>&1 | head -n 1)"
    echo "tesseract version: $(tesseract --version 2>&1 | head -n 1)"
    echo "java version: $(java -version 2>&1 | head -n 1)"
  '';
}

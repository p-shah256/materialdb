# shell.nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Core development tools
    gcc12
    gdb
    gnumake

    timg  # Add this line
    # OpenCV and its dependencies
    opencv4

    # LSP server and tools for C++
    clang-tools # Provides clangd
    ccls       # Alternative LSP server

    # Build system essentials
    pkg-config

    # Additional development tools
    bear      # For generating compilation database
  ];

  # Environment variables
  shellHook = ''
    # Set up pkg-config path to find OpenCV
    export PKG_CONFIG_PATH="${pkgs.opencv4}/lib/pkgconfig:$PKG_CONFIG_PATH"
    # Create compile_commands.json for LSP
    echo '[
      {
        "directory": "'"$PWD"'",
        "command": "g++ -std=c++17 -I${pkgs.opencv4}/include/opencv4 -c TableDetector.cpp -o TableDetector.o",
        "file": "TableDetector.cpp"
      },
      {
        "directory": "'"$PWD"'",
        "command": "g++ -std=c++17 -I${pkgs.opencv4}/include/opencv4 -c main.cpp -o main.o",
        "file": "main.cpp"
      }
    ]' > compile_commands.json

    # Print helpful information
    echo "OpenCV Development Environment Ready!"
    echo "OpenCV include path: ${pkgs.opencv4}/include/opencv4"
    echo "LSP servers available: clangd and ccls"
  '';
}

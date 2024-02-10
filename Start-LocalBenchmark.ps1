Write-Host "[Compile]"
cargo build --release
Move-Item ../target/release/ahc030.exe . -Force
Write-Host "[Run]"
$env:DURATION_MUL = "1.5"
dotnet marathon run-local
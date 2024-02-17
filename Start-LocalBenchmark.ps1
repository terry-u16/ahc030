Write-Host "[Compile]"
cargo build --release
Move-Item ../target/release/ahc030.exe . -Force
Write-Host "[Run]"
$env:DURATION_MUL = "1.5"
$env:AHC030_SHOW_COMMENT = 0
dotnet marathon run-local
./relative_score.exe -d ./data/results -o min
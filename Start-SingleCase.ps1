param(
    [Parameter(mandatory)]
    [int]
    $seed
)

$in = ".\data\in\{0:0000}.txt" -f $seed
$env:DURATION_MUL = "1.5"
Get-Content $in | .\tester.exe cargo run --release > .\out.txt
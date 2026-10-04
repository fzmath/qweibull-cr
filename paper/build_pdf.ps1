Set-Location "G:\OPT\q-Weibull竞争风险论文\paper"
pdflatex -interaction=nonstopmode main.tex | Out-Null
bibtex main | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
if (Test-Path main.pdf) {
  $f = Get-Item main.pdf
  Write-Output ("BUILD DONE; main.pdf {0} bytes; modified {1}" -f $f.Length, $f.LastWriteTime)
} else {
  Write-Output "BUILD FAILED: no pdf"
}

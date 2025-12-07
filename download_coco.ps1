# download_coco.ps1
# Script untuk download & extract COCO 2017 ke D:\datasets\coco2017

$CocoRoot = "D:\datasets\coco2017"

Write-Host "Membuat folder $CocoRoot (jika belum ada)..."
New-Item -ItemType Directory -Path $CocoRoot -Force | Out-Null
Set-Location $CocoRoot

# Daftar file COCO 2017
$files = @{
    "train2017.zip"                 = "http://images.cocodataset.org/zips/train2017.zip";
    "val2017.zip"                   = "http://images.cocodataset.org/zips/val2017.zip";
    "test2017.zip"                  = "http://images.cocodataset.org/zips/test2017.zip";
    "annotations_trainval2017.zip"  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

Write-Host "Mulai download file COCO 2017..."
foreach ($name in $files.Keys) {
    $url  = $files[$name]
    $dest = Join-Path $CocoRoot $name

    if (Test-Path $dest) {
        Write-Host "$name sudah ada, skip download."
    } else {
        Write-Host "Downloading $name ..."
        Invoke-WebRequest -Uri $url -OutFile $dest
    }
}

Write-Host "Extract semua ZIP..."
if (Test-Path "$CocoRoot\train2017.zip") {
    Write-Host "Extracting train2017.zip ..."
    Expand-Archive "$CocoRoot\train2017.zip" -DestinationPath $CocoRoot -Force
}
if (Test-Path "$CocoRoot\val2017.zip") {
    Write-Host "Extracting val2017.zip ..."
    Expand-Archive "$CocoRoot\val2017.zip" -DestinationPath $CocoRoot -Force
}
if (Test-Path "$CocoRoot\test2017.zip") {
    Write-Host "Extracting test2017.zip ..."
    Expand-Archive "$CocoRoot\test2017.zip" -DestinationPath $CocoRoot -Force
}
if (Test-Path "$CocoRoot\annotations_trainval2017.zip") {
    Write-Host "Extracting annotations_trainval2017.zip ..."
    Expand-Archive "$CocoRoot\annotations_trainval2017.zip" -DestinationPath $CocoRoot -Force
}

Write-Host "Hapus file .zip untuk hemat space..."
Get-ChildItem "$CocoRoot\*.zip" -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host "SELESAI: COCO 2017 sekarang ada di $CocoRoot"

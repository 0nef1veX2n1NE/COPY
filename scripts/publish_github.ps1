param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$Repository,

    [ValidateSet('private', 'public')]
    [string]$Visibility = 'private',

    [string]$Owner = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $ProjectRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git이 없습니다. https://git-scm.com/download/win 에서 설치한 뒤 다시 실행하세요.'
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'GitHub CLI가 없습니다. winget install --id GitHub.cli 후 gh auth login을 실행하세요.'
}

if (-not (Test-Path (Join-Path $ProjectRoot '.git'))) {
    git init -b main
}
git add .
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m 'Set up chord estimation web service for Render'
}

$RemoteName = if ($Owner) { "$Owner/$Repository" } else { $Repository }
if (-not (git remote get-url origin 2>$null)) {
    gh repo create $RemoteName --$Visibility --source . --remote origin --push
} else {
    git push -u origin main
}

Write-Host "GitHub 업로드 완료: $RemoteName"
Write-Host '다음: https://dashboard.render.com/blueprints 에서 이 저장소의 render.yaml을 적용하세요.'

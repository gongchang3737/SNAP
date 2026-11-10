$ErrorActionPreference = "Stop"

$Python = "python"
$ResultDir = "result\pmc_sgbs"
$Seeds = @(
    47001, 48001, 49001, 50001, 51001,
    52001, 53001, 54001, 55001, 56001,
    57001, 58001, 59001, 60001, 61001,
    62001, 63001, 64001, 65001, 66001
)
$Cases = @(
    @("BA", 20, 3), @("BA", 50, 5), @("BA", 100, 10),
    @("ER", 20, 3), @("ER", 50, 5), @("ER", 100, 10),
    @("WS", 20, 3), @("WS", 50, 5), @("WS", 100, 10)
)

New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

function Test-SeedComplete {
    param([int]$SearchSeed)

    foreach ($Case in $Cases) {
        $Graph = $Case[0]
        $Size = $Case[1]
        $Budget = $Case[2]
        $Name = "ppo_mix_v3_sd_${Graph}_${Size}_k${Budget}_pmc_all_beta4_gamma4_M12_searchseed${SearchSeed}.json"
        $Path = Join-Path $ResultDir $Name
        if (-not (Test-Path -LiteralPath $Path)) {
            return $false
        }
        try {
            $Json = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
            $RecordedSeed = [int]$Json.PPOAgent.SGBS.search_config.seed
        }
        catch {
            return $false
        }
        if ($RecordedSeed -ne $SearchSeed) {
            return $false
        }
    }
    return $true
}

$Total = $Seeds.Count
for ($Index = 0; $Index -lt $Total; $Index++) {
    $SearchSeed = $Seeds[$Index]
    $Ordinal = $Index + 1

    if (Test-SeedComplete -SearchSeed $SearchSeed) {
        Write-Host "[$Ordinal/$Total] search seed=$SearchSeed already complete; skipping" -ForegroundColor DarkYellow
        continue
    }

    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "[$Ordinal/$Total] starting pMC-SGBS, search seed=$SearchSeed" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $ResultTag = "pmc_all_beta4_gamma4_M12_searchseed${SearchSeed}"
    $CliArgs = @(
        "run_experiment.py", "test",
        "--model-id", "ppo_mix_v3_sd",
        "--model-method", "PPOAgent",
        "--graph-types", "BA,ER,WS",
        "--test-sizes", "20,50,100",
        "--num-graphs", "100",
        "--beta", "4",
        "--gamma", "4",
        "--no-sd", "False",
        "--no-baselines",
        "--neural-methods", "SGBS",
        "--seed", "42",
        "--feature-set", "f3",
        "--sd-score-mode", "pmc",
        "--known-sd-level", "all",
        "--search-scenarios", "12",
        "--search-seed", "$SearchSeed",
        "--delay-mu", "1.0",
        "--results-dir", $ResultDir,
        "--result-tag", $ResultTag
    )

    & $Python -u @CliArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pMC-SGBS failed for search seed=$SearchSeed (exit code $LASTEXITCODE)"
    }
    if (-not (Test-SeedComplete -SearchSeed $SearchSeed)) {
        throw "search seed=$SearchSeed finished, but the nine result files failed validation"
    }

    Write-Host "[$Ordinal/$Total] search seed=$SearchSeed complete and validated" -ForegroundColor Green
}

Write-Host "`nAll 20 pMC-SGBS search-seed experiments are complete" -ForegroundColor Green

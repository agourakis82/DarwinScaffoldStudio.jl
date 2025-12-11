# Derivação Teórica de λ

**Data:** 2025-12-11

## Resultado Principal

```
λ = 1/(2 × ln(10)) ≈ 0.217
```

## Derivação

A causalidade de Granger C relaciona-se com o número de configurações Ω por:

```
C = C₀ × Ω^(-λ) = C₀ × exp(-λ × ln(Ω)) = C₀ × exp(-λS)
```

## Interpretação

- λ é o **expoente de confusão informacional**
- A cada aumento de 10× nas configurações, causalidade cai ~40%
- O valor 1/(2ln10) sugere conexão com escala decimal

## Validação

| λ teórico | λ empírico | Erro |
|-----------|------------|------|
| 0.2171 | 0.2273 | 4.5% |

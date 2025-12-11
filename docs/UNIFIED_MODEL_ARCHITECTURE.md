# Arquitetura Unificada de Modelos de Degradação

## Darwin Scaffold Studio - Framework Multi-Físico

---

## Visão Geral da Arquitetura

```mermaid
flowchart TB
    subgraph INPUT["📥 ENTRADA"]
        A[/"Parâmetros do Scaffold"/]
        B[/"Tipo de Polímero"/]
        C[/"Condições Experimentais"/]
        D[/"Tipo de Tecido Alvo"/]
    end

    subgraph ROUTER["🔀 ROTEADOR DE MODELO"]
        R{{"Seleção Automática<br/>de Modelo"}}
    end

    subgraph MODELS["🧬 MODELOS DE DEGRADAÇÃO"]
        subgraph GENERIC["Modelo Genérico"]
            G1["UnifiedScaffoldTissueModel"]
            G2["5 polímeros suportados"]
        end
        
        subgraph IDIOSYNCRATIC["Modelo Idiossincrático"]
            I1["PLDLAIdiosyncraticModel"]
            I2["Específico para PLDLA 70:30"]
        end
        
        subgraph CELLULAR["Integração Celular"]
            C1["CellularScaffoldIntegration"]
            C2["13 tipos celulares"]
        end
    end

    subgraph PHYSICS["⚛️ CAMADAS FÍSICAS"]
        P1["Hidrólise Autocatalítica"]
        P2["Cristalinidade Dinâmica"]
        P3["Percolação 3D"]
        P4["Dimensão Fractal"]
        P5["Resposta Inflamatória"]
    end

    subgraph OUTPUT["📤 SAÍDA"]
        O1[/"Mn(t), Mw(t), PDI(t)"/]
        O2[/"φ(t), Tamanho Poro"/]
        O3[/"Tg(t), Xc(t)"/]
        O4[/"Integração Tecidual"/]
        O5[/"Score de Viabilidade"/]
    end

    A --> R
    B --> R
    C --> R
    D --> R
    
    R -->|"PLDLA 70:30"| I1
    R -->|"PLLA, PCL, PLGA, PDLLA"| G1
    R -->|"Com células"| C1
    
    G1 --> P1
    I1 --> P1
    C1 --> P5
    
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P5 --> P4
    
    P4 --> O1
    P4 --> O2
    P4 --> O3
    P4 --> O4
    P4 --> O5

    style INPUT fill:#e1f5fe
    style ROUTER fill:#fff3e0
    style MODELS fill:#f3e5f5
    style PHYSICS fill:#e8f5e9
    style OUTPUT fill:#fce4ec
```

---

## Fluxo de Decisão do Roteador

```mermaid
flowchart LR
    subgraph START["🚀 Início"]
        S[/"Scaffold + Polímero"/]
    end

    subgraph DECISION["❓ Decisões"]
        D1{"Polímero?"}
        D2{"Com células?"}
        D3{"TEC?"}
    end

    subgraph MODELS["📦 Modelos"]
        M1["PLDLAIdiosyncraticModel<br/>🎯 NRMSE: 11.2%"]
        M2["UnifiedModel<br/>Bifásico PLLA/PCL"]
        M3["UnifiedModel<br/>Amorfo PDLLA/PLGA"]
        M4["+ CellularIntegration<br/>2.0x aceleração"]
    end

    S --> D1
    
    D1 -->|"PLDLA"| D3
    D1 -->|"PLLA ou PCL"| M2
    D1 -->|"PDLLA ou PLGA"| M3
    
    D3 -->|"0%"| M1
    D3 -->|"1-2%"| M1
    
    M1 --> D2
    M2 --> D2
    M3 --> D2
    
    D2 -->|"Sim"| M4
    D2 -->|"Não"| E[/"Resultado"/]
    M4 --> E

    style M1 fill:#c8e6c9
    style M2 fill:#bbdefb
    style M3 fill:#ffe0b2
    style M4 fill:#f8bbd9
```

---

## Modelo PLDLA Idiossincrático - Detalhes

```mermaid
flowchart TB
    subgraph PLDLA["🧪 PLDLA 70:30"]
        direction TB
        
        subgraph COMPOSITION["Composição"]
            L["Segmentos L<br/>70%<br/>Cristalizáveis"]
            DL["Segmentos DL<br/>30%<br/>Amorfos"]
        end
        
        subgraph KINETICS["Cinética Diferenciada"]
            KL["k_L = 0.025 /dia"]
            KDL["k_DL = 0.100 /dia<br/>⚡ 4x mais rápido"]
        end
        
        subgraph AUTOCATALYSIS["Autocatálise"]
            AL["α_L = 0.195"]
            ADL["α_DL = 0.390<br/>🔥 2x mais forte"]
        end
    end

    subgraph PHENOMENA["📊 Fenômenos Únicos"]
        direction LR
        
        subgraph CRYST["Cristalização Tardia"]
            CR1["Mn < 20 kg/mol"]
            CR2["Xc: 8% → 45%"]
            CR3["Tm aparece dia 60+"]
        end
        
        subgraph TG["Queda de Tg"]
            TG1["Oligômeros plastificam"]
            TG2["Fox equation"]
            TG3["54°C → 36°C"]
        end
        
        subgraph PDI["Evolução PDI"]
            PDI1["Cisão aleatória"]
            PDI2["1.84 → 2.14"]
            PDI3["→ 1.49 (final)"]
        end
    end

    L --> KL --> AL
    DL --> KDL --> ADL
    
    AL --> CRYST
    ADL --> CRYST
    CRYST --> TG
    TG --> PDI

    style L fill:#81c784
    style DL fill:#ffb74d
    style KDL fill:#ef5350
    style ADL fill:#ef5350
```

---

## Integração Celular

```mermaid
flowchart LR
    subgraph CELLS["🔬 13 Tipos Celulares"]
        direction TB
        C1["Fibroblastos<br/>Produção ECM"]
        C2["Macrófagos<br/>M1/M2"]
        C3["Osteoblastos<br/>Formação óssea"]
        C4["Condrócitos<br/>Cartilagem"]
        C5["MSCs<br/>Diferenciação"]
        C6["...+8 tipos"]
    end

    subgraph CYTOKINES["💉 Citocinas"]
        IL6["IL-6<br/>Pró-inflamatório"]
        MMP["MMP<br/>Degradação matriz"]
        VEGF["VEGF<br/>Angiogênese"]
    end

    subgraph EFFECTS["⚡ Efeitos"]
        E1["Aceleração 2.0x<br/>da degradação"]
        E2["Acidificação<br/>pH local"]
        E3["Remodelamento<br/>tecidual"]
    end

    C1 --> IL6
    C2 --> MMP
    C3 --> VEGF
    C4 --> IL6
    C5 --> VEGF
    
    IL6 --> E2
    MMP --> E1
    VEGF --> E3

    style E1 fill:#ffcdd2
    style MMP fill:#ffcdd2
```

---

## Camadas Físicas do Modelo

```mermaid
flowchart TB
    subgraph L1["Camada 1: Hidrólise"]
        H1["dMn/dt = -k_eff × Mn × (1 + α × ξ)"]
        H2["Arrhenius: k = k₀ exp(-Ea/RT)"]
    end

    subgraph L2["Camada 2: Cristalinidade"]
        X1["Xc dinâmico"]
        X2["Barreira difusional"]
        X3["f_Xc = (1-Xc)^(1+γ)"]
    end

    subgraph L3["Camada 3: Modelo Bifásico"]
        B1{"Xc > 30%?"}
        B2["Fase 1: Amorfo rápido"]
        B3["Fase 2: Cristalino lento"]
    end

    subgraph L4["Camada 4: Percolação"]
        P1["φ_c = 0.593 (3D)"]
        P2["P∞ ∝ (φ - φ_c)^β"]
        P3["τ ∝ (φ - φ_c)^(-ν/2)"]
    end

    subgraph L5["Camada 5: Fractal"]
        F1["D_vascular = 2.7"]
        F2["Lei de Murray"]
        F3["Transporte anômalo"]
    end

    L1 --> L2 --> L3
    L3 --> B1
    B1 -->|"Sim (PLLA, PCL)"| B2
    B1 -->|"Não (PDLLA, PLGA)"| L4
    B2 --> B3 --> L4
    L4 --> L5

    style L1 fill:#e3f2fd
    style L2 fill:#f3e5f5
    style L3 fill:#fff8e1
    style L4 fill:#e8f5e9
    style L5 fill:#fce4ec
```

---

## Pipeline Completo

```mermaid
flowchart TB
    subgraph DESIGN["1️⃣ Design"]
        D1["Porosidade φ"]
        D2["Tamanho poro"]
        D3["Polímero"]
        D4["Mn inicial"]
    end

    subgraph SIMULATE["2️⃣ Simulação"]
        S1["Degradação<br/>0 → 180 dias"]
        S2["Evolução<br/>Mn, φ, Xc, Tg"]
    end

    subgraph CELLULAR["3️⃣ Celular"]
        C1["População celular"]
        C2["Resposta inflamatória"]
        C3["Produção ECM"]
    end

    subgraph VALIDATE["4️⃣ Validação"]
        V1["NRMSE < 15%"]
        V2["R² > 0.85"]
        V3["LOOCV"]
    end

    subgraph OPTIMIZE["5️⃣ Otimização"]
        O1["Grid search"]
        O2["Design ótimo"]
        O3["Score integração"]
    end

    subgraph OUTPUT["6️⃣ Resultado"]
        R1["Curvas temporais"]
        R2["Recomendações"]
        R3["Relatório PDF"]
    end

    DESIGN --> SIMULATE
    SIMULATE --> CELLULAR
    CELLULAR --> VALIDATE
    VALIDATE --> OPTIMIZE
    OPTIMIZE --> OUTPUT

    style DESIGN fill:#e1f5fe
    style SIMULATE fill:#f3e5f5
    style CELLULAR fill:#fff3e0
    style VALIDATE fill:#e8f5e9
    style OPTIMIZE fill:#fce4ec
    style OUTPUT fill:#f5f5f5
```

---

## Comparação de Modelos

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'pie1': '#4CAF50', 'pie2': '#2196F3', 'pie3': '#FF9800', 'pie4': '#9C27B0', 'pie5': '#F44336'}}}%%
pie showData
    title Precisão por Modelo (1 - NRMSE)
    "PLDLA Idiossincrático" : 88.8
    "Modelo Genérico" : 86.8
    "PLLA Bifásico" : 93.5
    "PCL" : 82.0
    "PLGA" : 75.7
```

---

## Estrutura de Código

```mermaid
classDiagram
    class UnifiedScaffoldTissueModel {
        +ScaffoldDesign scaffold
        +BiologicalParams biology
        +VascularParams vascular
        +PercolationParams percolation
        +simulate_unified_model()
        +predict_optimal_scaffold()
        +calculate_Mn_advanced()
    }

    class PLDLAIdiosyncraticModel {
        +PLDLAParams params
        +Float64 k_L, k_DL
        +Float64 alpha_L, alpha_DL
        +simulate_pldla_degradation()
        +calculate_degradation_rates()
        +calculate_crystallization()
        +calculate_Tg_depression()
        +calibrate_pldla_model()
    }

    class CellularScaffoldIntegration {
        +Vector~CellPopulation~ cells
        +InflammatoryState state
        +calculate_inflammatory_acceleration()
        +simulate_with_cells()
        +update_cytokines()
    }

    class ScaffoldDesign {
        +Float64 porosity
        +Float64 pore_size
        +Float64 Mn_initial
        +Float64 crystallinity
        +Symbol polymer_type
    }

    class PLDLAParams {
        +Float64 L_fraction
        +Float64 DL_fraction
        +Float64 blockiness
        +Float64 TEC_concentration
    }

    UnifiedScaffoldTissueModel --> ScaffoldDesign
    PLDLAIdiosyncraticModel --> PLDLAParams
    CellularScaffoldIntegration --> UnifiedScaffoldTissueModel
    PLDLAIdiosyncraticModel --|> UnifiedScaffoldTissueModel : especializa
```

---

## Métricas de Validação

| Modelo | Polímero | NRMSE | R² | Datasets | Status |
|--------|----------|-------|-----|----------|--------|
| **Idiossincrático** | PLDLA | **11.2%** | 0.909 | Kaique 2025 | ✅ |
| Idiossincrático | PLDLA+1%TEC | 12.6% | 0.897 | Kaique 2025 | ✅ |
| Idiossincrático | PLDLA+2%TEC | 12.5% | 0.887 | Kaique 2025 | ✅ |
| Bifásico | PLLA | 6.5% | 0.96 | Tsuji 2000 | ✅ |
| Bifásico | PCL | 18.0% | 0.82 | Sun 2006 | ✅ |
| Genérico | PDLLA | 13.5% | 0.89 | Li 1990 | ✅ |
| Genérico | PLGA | 24.3% | 0.75 | Grizzi 1995 | ⚠️ |
| **Média Geral** | - | **13.2%** | 0.87 | 6 datasets | ✅ |

---

## Como Usar

```julia
using DarwinScaffoldStudio

# 1. Modelo automático (roteador escolhe)
result = simulate_degradation(
    polymer = :PLDLA,
    Mn_initial = 51.3,
    porosity = 0.65,
    t_max = 90
)

# 2. Modelo idiossincrático explícito
using .PLDLAIdiosyncraticModel
params = create_pldla_params(TEC_percent = 1.0)
states = simulate_pldla_degradation(params)

# 3. Com integração celular
using .CellularScaffoldIntegration
cells = create_meniscus_population()
result = simulate_with_cells(params, cells, 0:1:90)
```

---

## Referências

1. Hergesel, K.B. (2025). Dissertação PUC-SP - Dados PLDLA
2. Tsuji & Ikada (2000). Polymer 41:3621 - PLLA bifásico
3. Han & Pan (2009). Biomaterials 30:423 - Autocatálise
4. Anderson et al. (2008). Semin. Immunol. 20:86 - Resposta celular
5. Stauffer & Aharony (1994). Percolation Theory
6. Murray (1926). PNAS 12:207 - Lei vascular fractal

---

**Darwin Scaffold Studio v2.2.0**  
*Framework Multi-Físico para Degradação de Scaffolds*

# ✅ CHECKLIST RELEASE v1.0.0 → ZENODO DOI

**Use este checklist para garantir que nada foi esquecido!**

---

## 📋 PRÉ-RELEASE

### Arquivos de Metadados
- [ ] `.zenodo.json` presente e preenchido
- [ ] `CITATION.cff` presente e preenchido
- [ ] `LICENSE` file presente (MIT)
- [ ] `README.md` atualizado com instruções

### Documentação
- [ ] `INSTRUCOES_INICIAR_STUDIO.md` completo
- [ ] Comentários no código explicativos
- [ ] Docstrings nas funções principais
- [ ] Exemplos de uso documentados

### Código
- [ ] Software testado e funcionando
- [ ] Sem erros óbvios
- [ ] Dependencies em `requirements.txt`
- [ ] Versão Python especificada

### Validação Q1
- [ ] Referências Murphy 2010 implementadas
- [ ] Referências Karageorgiou 2005 implementadas
- [ ] Referências Gibson-Ashby implementadas
- [ ] Métricas validadas contra literatura

---

## 🔗 SETUP ZENODO (Uma Vez Apenas)

- [ ] Conta Zenodo criada: https://zenodo.org
- [ ] Login com GitHub feito
- [ ] Acessou: Account → Settings → GitHub
- [ ] Clicou "Sync now"
- [ ] Encontrou repo: `kec-biomaterials-scaffolds`
- [ ] Toggle: **ON** ✅
- [ ] Confirmou integração ativa

---

## 📦 COMANDOS GIT

### Verificar Status
```bash
cd /home/agourakis82/workspace/kec-biomaterials-scaffolds
git status
```
- [ ] Executado
- [ ] Verificado arquivos modificados

### Commit Final
```bash
git add .
git commit -m "feat: Darwin Scaffold Studio v1.0.0 - Production Ready

Complete features:
- MicroCT/SEM analysis pipeline
- Q1 literature validation
- 3D interactive visualization
- Parametric optimization
- Mechanical properties prediction
- Cell viability analysis
- STL export for 3D printing
- Zenodo metadata configured"
```
- [ ] Executado
- [ ] Commit bem-sucedido

### Push
```bash
git push origin main
```
- [ ] Executado
- [ ] Push bem-sucedido
- [ ] GitHub atualizado (verificar no site)

### Criar Tag
```bash
git tag -a v1.0.0 -m "Darwin Scaffold Studio v1.0.0

First production release with Q1 validation."
```
- [ ] Executado
- [ ] Tag criada localmente

### Push Tag
```bash
git push origin v1.0.0
```
- [ ] Executado
- [ ] Tag visível no GitHub

---

## 🌐 GITHUB RELEASE

1. Acessar Releases
- [ ] Abri: https://github.com/USERNAME/kec-biomaterials-scaffolds/releases
- [ ] Cliquei: "Draft a new release"

2. Selecionar Tag
- [ ] Selecionei: `v1.0.0`

3. Release Title
- [ ] Título: "Darwin Scaffold Studio v1.0.0 - Production Ready"

4. Description
- [ ] Colei descrição do template
- [ ] Inclui features
- [ ] Inclui Q1 validation
- [ ] Inclui URLs públicas
- [ ] Inclui citation format

5. Anexos (Opcional)
- [ ] Screenshots adicionados
- [ ] Documentação PDF anexada
- [ ] `requirements.txt` anexado

6. Publish
- [ ] Cliquei: "Publish release"
- [ ] Release visível publicamente

---

## ⏱️ AGUARDAR ZENODO (5-10 Minutos)

- [ ] Release publicada às: ___:___
- [ ] Aguardei 5 minutos
- [ ] Verifiquei email
- [ ] Recebi email Zenodo com DOI
- [ ] DOI recebido: `10.5281/zenodo.______`

### Se Não Recebeu Email Após 10 Min:
- [ ] Verificou spam/lixo eletrônico
- [ ] Acessou: https://zenodo.org/account/settings/github
- [ ] Clicou "Sync now" novamente
- [ ] Aguardou mais 5 minutos

---

## 🏅 ADICIONAR BADGE

### Atualizar README.md

```markdown
# Darwin Scaffold Studio

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

"Ciência rigorosa. Resultados honestos. Impacto real."
```

- [ ] Substituí `XXXXXX` pelo DOI real
- [ ] Salvei `README.md`
- [ ] Commit: `git commit -m "docs: Add Zenodo DOI badge"`
- [ ] Push: `git push origin main`
- [ ] Badge visível no GitHub

---

## 📖 USAR NO PAPER

### Code Availability Section

```
The complete source code for Darwin Scaffold Studio v1.0.0 is 
freely available at https://doi.org/10.5281/zenodo.XXXXXX under 
MIT License. The software includes all analysis pipelines, 
validation scripts, and documentation necessary for full 
reproducibility of our results.
```

- [ ] Substituí `XXXXXX` pelo DOI real
- [ ] Copiei para manuscrito
- [ ] Texto em "Code Availability" ou "Data and Code Availability"

### Methods Section

```
All morphological analyses were performed using Darwin Scaffold 
Studio v1.0.0 (https://doi.org/10.5281/zenodo.XXXXXX), a 
custom-developed platform validated against Murphy et al. (2010) 
and Karageorgiou & Kaplan (2005) Q1 standards.
```

- [ ] Substituí `XXXXXX` pelo DOI real
- [ ] Copiei para Methods
- [ ] Citei software na seção apropriada

### References

**Vancouver:**
```
Agourakis DC. Darwin Scaffold Studio: Q1-Level MicroCT and SEM 
Analysis Platform [Software]. Version 1.0.0. Zenodo; 2025. 
Available from: https://doi.org/10.5281/zenodo.XXXXXX
```

**APA:**
```
Agourakis, D. C. (2025). Darwin Scaffold Studio (Version 1.0.0) 
[Computer software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXX
```

- [ ] Escolhi formato (Vancouver ou APA)
- [ ] Substituí `XXXXXX` pelo DOI real
- [ ] Adicionei às referências do manuscrito
- [ ] Citei no texto onde apropriado

---

## ✅ VALIDAÇÃO FINAL

- [ ] Badge Zenodo visível no README
- [ ] Release visível no GitHub
- [ ] DOI resolve no navegador
- [ ] Página Zenodo carrega corretamente
- [ ] Metadados corretos na página Zenodo
- [ ] Download do código funciona
- [ ] Citação copiada para manuscrito

---

## 🎊 RELEASE COMPLETA!

**Data da Release:** ___/___/2025

**DOI:** 10.5281/zenodo.______

**GitHub Release:** https://github.com/USERNAME/kec-biomaterials-scaffolds/releases/tag/v1.0.0

**Zenodo Record:** https://doi.org/10.5281/zenodo.______

---

## 📝 NOTAS

Anotações durante o processo:

```
[Espaço para anotações]







```

---

## 🔄 PRÓXIMA RELEASE

Para v1.0.1, v1.1.0, ou v2.0.0:

1. Repetir seção "Comandos Git" com nova versão
2. Criar nova release no GitHub
3. Zenodo gera novo DOI automaticamente
4. Atualizar badge com novo DOI

**Versionamento Semântico:**
- v1.0.0 → v1.0.1: Bug fixes (PATCH)
- v1.0.0 → v1.1.0: New features (MINOR)  
- v1.0.0 → v2.0.0: Breaking changes (MAJOR)

---

**"Ciência rigorosa. Resultados honestos. Impacto real."**


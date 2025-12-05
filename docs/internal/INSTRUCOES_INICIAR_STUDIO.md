# 🚀 INSTRUÇÕES PARA INICIAR DARWIN SCAFFOLD STUDIO

Dr. Agourakis,

**O terminal está com problemas, então por favor execute MANUALMENTE:**

---

## ⚡ OPÇÃO 1: Comando Direto (RECOMENDADO)

Abra um terminal e execute:

```bash
cd /home/agourakis82/workspace/kec-biomaterials-scaffolds

# Parar processos antigos
pkill -f "streamlit.*8600"

# Esperar 3 segundos
sleep 3

# Iniciar Streamlit
streamlit run apps/production/darwin_scaffold_studio.py --server.port 8600 --server.headless true
```

**Aguarde a mensagem:**
```
You can now view your Streamlit app in your browser.
URL: http://localhost:8600
```

---

## ⚡ OPÇÃO 2: Script Automático

```bash
cd /home/agourakis82/workspace/kec-biomaterials-scaffolds
chmod +x START_STUDIO.sh
./START_STUDIO.sh
```

---

## 🌐 DEPOIS DE INICIAR:

1. Aguarde ~10 segundos
2. Abra navegador
3. Acesse: **http://localhost:8600/**
4. Se aparecer "connection refused", aguarde mais um pouco

---

## ✅ ESTÁ FUNCIONANDO QUANDO:

Você vê no terminal:
```
  You can now view your Streamlit app in your browser.

  URL: http://localhost:8600
```

---

## 🐛 SE DER ERRO:

**Copie a mensagem de erro completa** e me envie!

Erros comuns:
- **Port already in use** → Execute: `pkill -f streamlit` e tente novamente
- **Module not found** → Ambiente errado, ative o correto
- **Syntax error no código** → Me envie o erro completo

---

## 🎨 O QUE ESPERAR:

Quando funcionar, você verá:

✅ **STAGE 1: Upload**
- Upload de arquivo ou Demo Dataset

✅ **STAGE 2: Analyze**
- 5 steps com progress bar
- RAW image (2D ou 3D com slider!)
- Mechanical properties Gibson-Ashby

✅ **STAGE 3: Optimize**
- 3D mesh INTERATIVO Plotly (arraste, zoom!)
- Heatmap INTERATIVO Plotly (hover detalhes!)
- Mechanical properties antes de otimizar
- Configure targets e gere otimizado

✅ **STAGE 4: Preview 3D**
- 2 meshes 3D lado a lado (Original vs Optimized!)
- Comparison table COM mechanical properties!
- Improvements %

✅ **STAGE 5: Export STL**
- Download 2 STL files (original + optimized!)
- Download JSON report
- Pronto para impressão 3D!

---

**EXECUTE UM DOS COMANDOS ACIMA E ME AVISE SE FUNCIONOU! 🚀**


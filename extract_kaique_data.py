#!/usr/bin/env python3
"""Extract experimental data from Kaique thesis for model calibration."""

import re

import fitz


def extract_text_with_tables(pdf_path):
    """Extract text from PDF looking for numerical data"""
    doc = fitz.open(pdf_path)

    results = {}
    results["mn_data"] = []
    results["porosity_data"] = []
    results["pore_size_data"] = []
    results["mechanical_data"] = []
    results["degradation_times"] = []

    full_text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text += f"\n--- PAGE {page_num + 1} ---\n{text}"

        # Mn patterns
        mn_matches = re.findall(r"[Mm]n\s*[=:]\s*(\d+[\.,]?\d*)", text)
        for m in mn_matches:
            results["mn_data"].append((page_num + 1, m))

        # Porosity patterns
        por_matches = re.findall(r"porosidade[^0-9]*(\d+[\.,]?\d*)\s*%", text, re.I)
        for m in por_matches:
            results["porosity_data"].append((page_num + 1, m))
        por_matches2 = re.findall(
            r"(\d+[\.,]?\d*)\s*%\s*(?:de\s+)?porosidade", text, re.I
        )
        for m in por_matches2:
            results["porosity_data"].append((page_num + 1, m))

        # Pore size patterns
        pore_matches = re.findall(r"(\d+[\.,]?\d*)\s*(?:μm|um|micr)", text, re.I)
        for m in pore_matches:
            results["pore_size_data"].append((page_num + 1, m))

        # Degradation times
        time_matches = re.findall(r"(\d+)\s*dias?", text, re.I)
        for m in time_matches:
            if int(m) < 200:
                results["degradation_times"].append((page_num + 1, m))

        # Mechanical data (MPa)
        mech_matches = re.findall(r"(\d+[\.,]?\d*)\s*(?:MPa|GPa)", text, re.I)
        for m in mech_matches:
            results["mechanical_data"].append((page_num + 1, m))

    doc.close()
    return results, full_text


# Main
import os

pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf") and "Kaique" in f]
pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf') and 'Kaique' in f]
pdf_path = pdf_files[0] if pdf_files else None

print("=" * 80)
print("  EXTRAÇÃO DE DADOS EXPERIMENTAIS - TESE DO KAIQUE")
print("=" * 80)

try:
    results, full_text = extract_text_with_tables(pdf_path)

    print("\n[Mn] DADOS DE MASSA MOLECULAR:")
    print("-" * 40)
    if results["mn_data"]:
        seen = set()
        for page, value in results["mn_data"]:
            if value not in seen:
                print(f"  Pagina {page}: Mn = {value}")
                seen.add(value)
    else:
        print("  Nenhum dado explicito encontrado")

    print("\n[PHI] DADOS DE POROSIDADE:")
    print("-" * 40)
    if results["porosity_data"]:
        seen = set()
        for page, value in results["porosity_data"]:
            if value not in seen:
                print(f"  Pagina {page}: {value}%")
                seen.add(value)
    else:
        print("  Nenhum dado explicito encontrado")

    print("\n[PORO] DADOS DE TAMANHO DE PORO:")
    print("-" * 40)
    if results["pore_size_data"]:
        seen = set()
        for page, value in results["pore_size_data"]:
            if value not in seen and float(value.replace(",", ".")) > 10:
                print(f"  Pagina {page}: {value} um")
                seen.add(value)
    else:
        print("  Nenhum dado explicito encontrado")

    print("\n[t] TEMPOS DE DEGRADACAO MENCIONADOS:")
    print("-" * 40)
    times = sorted(set([int(t[1]) for t in results["degradation_times"]]))
    times = [t for t in times if t > 0]
    print(f"  Tempos: {times[:15]} dias")

    print("\n[E] DADOS MECANICOS:")
    print("-" * 40)
    if results["mechanical_data"]:
        seen = set()
        for page, value in results["mechanical_data"]:
            if value not in seen:
                print(f"  Pagina {page}: {value} MPa")
                seen.add(value)
    else:
        print("  Nenhum dado explicito encontrado")

    # Save full text
    with open("kaique_thesis_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"\nTexto completo salvo em: kaique_thesis_text.txt")

except Exception as e:
    print(f"Erro: {e}")
    import traceback

    traceback.print_exc()

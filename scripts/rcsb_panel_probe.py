import requests
cands = {
  "5HRZ": "designed homotrimer (baseline)",
  "1HSG": "HIV protease dimer",
  "2HHB": "hemoglobin (AU=4)",
  "1RUZ": "influenza HA (glyco trimer)",
  "4TVP": "HIV Env SOSIP (glyco trimer)",
  "1STM": "STMV virus (icosahedral)",
}
for pdb, note in cands.items():
    row = {"pdb": pdb}
    try:
        e = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}", timeout=10).json()
        i = e.get("rcsb_entry_info", {})
        row["chains"] = i.get("deposited_polymer_entity_instance_count")
        row["res"] = i.get("deposited_polymer_monomer_count")
        row["nonpoly"] = i.get("nonpolymer_bound_components")
    except Exception as ex:
        row["e"] = str(ex)[:40]
    try:
        a = requests.get(f"https://data.rcsb.org/rest/v1/core/assembly/{pdb}/1", timeout=10).json()
        s = a.get("rcsb_struct_symmetry") or []
        row["sym"] = [(x.get("symbol"), x.get("oligomeric_state")) for x in s]
        row["oligo"] = (a.get("pdbx_struct_assembly") or {}).get("oligomeric_count")
    except Exception as ex:
        row["ae"] = str(ex)[:40]
    print(f"{pdb} [{note}] chains={row.get('chains')} res={row.get('res')} "
          f"oligo={row.get('oligo')} sym={row.get('sym')} nonpoly={row.get('nonpoly')}")



def get_regions(name_analysis):
    if name_analysis == "Analysis 2021":
        REGIONS_PRED = {"Yemen": {0:["Split 4", "Sana'a"], 1:["Split 4", "Al Jawf"]}, 
                        "Syria": {0:["Split 4", "Lattakia"], 1:["Split 5", "Damascus"]}, 
                        "Mali": {0:["Split 2", "Mopti"], 1:["Split 3", "Sikasso"]}, 
                        "Nigeria": {0:["Split 4", "Yobe"], 1:["Split 2", "Adamawa"]}, 
                        "Cameroon": {0:["Split 3", "Adamawa"], 1:["Split 2", "Central"]}, 
                        "Burkina Faso": {0:["Split 4", "Boucle-Du-Mouhoun"], 1:["Split 5", "Centre-Ouest"]}}
        return REGIONS_PRED
    if name_analysis == "Analysis 2022":
        REGIONS_PRED = {"Yemen": {0:["Split 3", "Amanat Al Asimah"], 1:["Split 5", "Abyan"]}, 
                        "Syria": {0:["Split 3", "Damascus"], 1:["Split 4", "Tartous"]}, 
                        "Mali": {0:["Split 5", "Koulikoro"], 1:["Split 3", "Sikasso"]}, 
                        "Nigeria": {0:["Split 4", "Yobe"], 1:["Split 4", "Borno"]}, 
                        "Cameroon": {0:["Split 5", "Central"], 1:["Split 3", "North-West"]}, 
                        "Burkina Faso": {0:["Split 3", "Boucle-Du-Mouhoun"], 1:["Split 4", "Centre-Ouest"]}}
        return REGIONS_PRED
    if name_analysis == "Analysis 2022 (more splits)": # TODO
        REGIONS_PRED = {"Yemen": {0:["Split 4", "Sana'a"], 1:["Split 4", "Al Jawf"]}, 
                        "Syria": {0:["Split 4", "Lattakia"], 1:["Split 5", "Damascus"]}, 
                        "Mali": {0:["Split 2", "Mopti"], 1:["Split 3", "Sikasso"]}, 
                        "Nigeria": {0:["Split 4", "Yobe"], 1:["Split 2", "Adamawa"]}, 
                        "Cameroon": {0:["Split 3", "Adamawa"], 1:["Split 2", "Central"]}, 
                        "Burkina Faso": {0:["Split 4", "Boucle-Du-Mouhoun"], 1:["Split 5", "Centre-Ouest"]}}
        return REGIONS_PRED
    if name_analysis == "Analysis 2022 Same Time Length": # TODO
        REGIONS_PRED = {"Yemen": {0:["Split 4", "Sana'a"], 1:["Split 4", "Al Jawf"]}, 
                        "Syria": {0:["Split 4", "Lattakia"], 1:["Split 5", "Damascus"]}, 
                        "Mali": {0:["Split 2", "Mopti"], 1:["Split 3", "Sikasso"]}, 
                        "Nigeria": {0:["Split 4", "Yobe"], 1:["Split 2", "Adamawa"]}, 
                        "Cameroon": {0:["Split 3", "Adamawa"], 1:["Split 2", "Central"]}, 
                        "Burkina Faso": {0:["Split 4", "Boucle-Du-Mouhoun"], 1:["Split 5", "Centre-Ouest"]}}
        return REGIONS_PRED
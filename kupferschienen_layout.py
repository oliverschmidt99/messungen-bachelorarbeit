import ezdxf
from ezdxf import colors
from ezdxf.math import Vec2

def create_busbar_dxf(filename="kupferschienen_layout.dxf"):
    # Erstelle ein neues DXF-Dokument
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()

    # Layer definieren
    doc.layers.new(name='Sammelschienen_L1_L2_L3', dxfattribs={'color': colors.WHITE})
    doc.layers.new(name='Parallel_Rot', dxfattribs={'color': colors.RED})
    doc.layers.new(name='Dreieck_Gruen', dxfattribs={'color': colors.GREEN})
    doc.layers.new(name='Phase_Blau', dxfattribs={'color': colors.BLUE})
    doc.layers.new(name='Phase_Orange', dxfattribs={'color': 30}) # 30 entspricht einem Orange-Ton im ACI
    doc.layers.new(name='Text_Labels', dxfattribs={'color': colors.WHITE})
    doc.layers.new(name='Bemassung', dxfattribs={'color': colors.WHITE})

    # Einheitliche Y-Höhe für alle Bemassungen
    dim_y_level = 0

    # 1. Horizontale Sammelschienen (L1, L2, L3) zeichnen
    y_L1, y_L2, y_L3 = 200, 160, 120
    bar_width = 1000
    bar_height = 10
    
    for y_pos, label in zip([y_L1, y_L2, y_L3], ["L1", "L2", "L3"]):
        # Schiene als Rechteck
        msp.add_lwpolyline([
            (0, y_pos), (bar_width, y_pos), 
            (bar_width, y_pos + bar_height), (0, y_pos + bar_height), 
            (0, y_pos)
        ], dxfattribs={'layer': 'Sammelschienen_L1_L2_L3'})
        # Label L1, L2, L3
        msp.add_text(label, dxfattribs={'layer': 'Text_Labels', 'height': 8}).set_placement((-20, y_pos))

    # Hilfsfunktion zum Zeichnen vertikaler Abgänge
    def draw_vertical_drop(x_start, y_top, y_bottom, layer_name):
        msp.add_lwpolyline([
            (x_start, y_top), (x_start + 10, y_top),
            (x_start + 10, y_bottom), (x_start, y_bottom),
            (x_start, y_top)
        ], dxfattribs={'layer': layer_name})

    # 2. Linker Abschnitt (Rot) - Parallelanordnung
    x_parallel = 50
    draw_vertical_drop(x_parallel, y_L1 + bar_height, 20, 'Parallel_Rot')
    msp.add_text("Parallelanordnung", dxfattribs={'layer': 'Text_Labels', 'height': 5}).set_placement((x_parallel - 10, 10))

    # 3. Linker Abschnitt (Grün) - Dreiecksanordnung
    x_dreieck = 120
    draw_vertical_drop(x_dreieck, y_L1 + bar_height, 20, 'Dreieck_Gruen')
    msp.add_text("Dreiecksanordnung", dxfattribs={'layer': 'Text_Labels', 'height': 5}).set_placement((x_dreieck - 10, 10))

    # 4. Rechter Abschnitt (Blau) - Phasenmittelabstand 130 mm
    x_blue_start = 300
    for i in range(3):
        current_x = x_blue_start + (i * 130)
        draw_vertical_drop(current_x, y_L1 + bar_height, 20, 'Phase_Blau')
        
        # Bemassung hinzufügen (zwischen den Phasen)
        if i > 0:
            prev_x = x_blue_start + ((i - 1) * 130) + 5 # +5 für die Mitte der 10mm breiten Schiene
            curr_x = current_x + 5
            dim = msp.add_linear_dim(
                base=(prev_x, dim_y_level), 
                p1=(prev_x, 20), 
                p2=(curr_x, 20),
                dxfattribs={'layer': 'Bemassung'}
            )
            dim.render()

    # 5. Rechter Abschnitt (Orange) - Phasenmittelabstand 130 mm (wie gefordert geändert)
    x_orange_start = 750
    for i in range(3):
        current_x = x_orange_start + (i * 130)
        draw_vertical_drop(current_x, y_L1 + bar_height, 20, 'Phase_Orange')
        
        # Bemassung hinzufügen
        if i > 0:
            prev_x = x_orange_start + ((i - 1) * 130) + 5
            curr_x = current_x + 5
            dim = msp.add_linear_dim(
                base=(prev_x, dim_y_level), 
                p1=(prev_x, 20), 
                p2=(curr_x, 20),
                dxfattribs={'layer': 'Bemassung'}
            )
            dim.render()

    # Datei speichern
    doc.saveas(filename)
    print(f"DXF-Datei '{filename}' wurde erfolgreich erstellt.")

if __name__ == "__main__":
create_busbar_dxf()
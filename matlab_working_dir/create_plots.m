
%% Automatische Diagrammerstellung
clear; clc; close all;

try
    % Konfiguration
    filename = 'plot_data.csv';
    phases = {'L1', 'L2', 'L3'};
    limits_class = 0.5; 
    nennstrom = 2000;

    if ~isfile(filename)
        error('Datei plot_data.csv nicht gefunden!');
    end

    data = readtable(filename, 'Delimiter', ',');

    % Farben definieren
    hex2rgb = @(hex) sscanf(hex(2:end),'%2x%2x%2x',[1 3])/255;

    % Trompeten-Grenzwerte
    x_lims = [1, 5, 20, 100, 120];
    if limits_class == 0.2
        y_lims = [0.75, 0.35, 0.2, 0.2, 0.2];
    elseif limits_class == 0.5
        y_lims = [1.5, 1.5, 0.75, 0.5, 0.5];
    elseif limits_class == 1.0
        y_lims = [3.0, 1.5, 1.0, 1.0, 1.0];
    else
        y_lims = [1.5, 1.5, 0.75, 0.5, 0.5];
    end

    for i = 1:length(phases)
        p = phases{i};
        fprintf('Bearbeite Phase %s...\n', p);
        
        % Daten filtern
        rows = strcmp(data.phase, p);
        sub_data = data(rows, :);
        
        if isempty(sub_data)
            fprintf('Warnung: Keine Daten fuer Phase %s.\n', p);
            continue; 
        end
        
        f = figure('Visible', 'off', 'PaperType', 'A4', 'PaperOrientation', 'landscape');
        set(f, 'Color', 'w'); 
        set(f, 'Units', 'centimeters', 'Position', [0 0 29.7 21]);
        
        t = tiledlayout(f, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        % --- PLOT 1: Fehler ---
        ax1 = nexttile;
        hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
        set(ax1, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.8 0.8 0.8]);
        
        plot(ax1, x_lims, y_lims, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        plot(ax1, x_lims, -y_lims, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        
        % Eindeutige Kurven anhand der trace_id finden
        [unique_traces, ~, ~] = unique(sub_data.trace_id, 'stable');
        
        for k = 1:length(unique_traces)
            trace = unique_traces{k};
            trace_rows = strcmp(sub_data.trace_id, trace);
            d = sub_data(trace_rows, :);
            [~, sort_idx] = sort(d.target_load);
            d = d(sort_idx, :);
            
            % HIER WAR DER FEHLER: Wir nutzen jetzt die korrekten Spaltennamen der CSV
            col_rgb = hex2rgb(d.color_hex{1});
            
            % Unterstriche escapen
            leg_lbl = strrep(d.legend_name{1}, '_', '\_');
            
            plot(ax1, d.target_load, d.err_ratio, '-o', 'Color', col_rgb, ...
                'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', col_rgb, ...
                'DisplayName', leg_lbl);
        end
        
        title(ax1, sprintf('Fehlerverlauf - Phase %s (%d A)', p, nennstrom), 'Color', 'k');
        ylabel(ax1, 'Fehler [%]', 'Color', 'k');
        ylim(ax1, [- 1.5, 1.5]);
        xlim(ax1, [0, 125]);
        
        lgd = legend(ax1, 'Location', 'southoutside', 'Orientation', 'horizontal');
        set(lgd, 'TextColor', 'k', 'Color', 'w', 'Interpreter', 'tex');
        lgd.NumColumns = 2;
        
        % --- PLOT 2: StdAbw (Side-by-Side) ---
        ax2 = nexttile;
        hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
        set(ax2, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.8 0.8 0.8]);
        
        num_groups = length(unique_traces);
        total_group_width = 5; 
        single_bar_width = total_group_width / num_groups;
        if single_bar_width > 1.5; single_bar_width = 1.5; end
        
        for k = 1:length(unique_traces)
            trace = unique_traces{k};
            trace_rows = strcmp(sub_data.trace_id, trace);
            d = sub_data(trace_rows, :);
            [~, sort_idx] = sort(d.target_load);
            d = d(sort_idx, :);
            
            col_rgb = hex2rgb(d.color_hex{1});
            leg_lbl = strrep(d.legend_name{1}, '_', '\_');
            
            offset = (k - 1 - (num_groups - 1) / 2) * single_bar_width;
            x_pos = d.target_load + offset;
            
            bar(ax2, x_pos, d.err_std, 'FaceColor', col_rgb, 'EdgeColor', 'none', ...
                'FaceAlpha', 0.8, 'BarWidth', 0.1, 'DisplayName', leg_lbl);
        end
        
        title(ax2, sprintf('Standardabweichung - Phase %s (%d A)', p, nennstrom), 'Color', 'k');
        ylabel(ax2, 'StdAbw [%]', 'Color', 'k');
        xlabel(ax2, 'Last [% I_{Nenn}]', 'Color', 'k');
        xlim(ax2, [0, 125]);
        
        out_name = sprintf('Detail_%s_%dA.pdf', p, nennstrom);
        exportgraphics(f, out_name, 'ContentType', 'vector', 'BackgroundColor', 'w');
        close(f);
    end
catch ME
    disp('FEHLER:');
    disp(ME.message);
    exit(1); 
end
exit(0);

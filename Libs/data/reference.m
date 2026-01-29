clc % alles was geschrieben wurde wird gelöscht
clear all % alles auf dem Workspace löschen
% Einlesen der Versuchsdaten
load
ARROWS_PA8_2_1_13_11_2024_2024_11_13_14_41_10_Messung_Balken_ACBS_20241113_155434_com
plete0_4800Hz.MAT
%% Eingangsdaten
% Zeiten zuweisen
Zeit1 = Channel_1_Data;
Zeit2 = Channel_10_Data;
Zeit3 = Channel_19_Data;
% Temperaturen zuweisen
Temperatur_Umgebung = Channel_17_Data;
Temperatur_Unterseite = Channel_18_Data;
% Kraft und Weg zuweisen pro Laststempel
Kraft1 = Channel_2_Data;
Weg1 = Channel_3_Data;
Kraft2 = Channel_4_Data;
Weg2 = Channel_5_Data;
Kraft3 = Channel_6_Data;
Weg3 = Channel_7_Data;
Kraft4 = Channel_8_Data;
Weg4 = Channel_9_Data;
Kraft5 = Channel_20_Data;
Weg5 = Channel_21_Data;
Kraft6 = Channel_22_Data;
Weg6 = Channel_23_Data;
Kraft7 = Channel_24_Data;
Weg7 = Channel_25_Data;
Kraft8 = Channel_26_Data;
Weg8 = Channel_27_Data;
Kraft9 = Channel_11_Data;
Weg9 = Channel_12_Data;
Kraft10 = Channel_13_Data;
Kraft11 = Channel_15_Data;
Weg11 = Channel_16_Data;
% Definieren einer Matrix für die Position der Laststempel
LS = 75:35:425;
LS = LS';
%% Abspeichern aller Kräfte (1-11) und aller Wege (1-11) in ein Zellarray
% Abspeichern der Wege
Wege = cell(1, 11);
for i = 1:11
Weg = evalin('base',['Weg', num2str(i)]);
Wege{i} = Weg; % alle Wege in [mm]
end
% Abspeichern der Kräfte
Kraefte = cell(1,11);
for i = 1:11
Kraft = evalin('base',['Kraft', num2str(i)]);
Kraefte{i} = Kraft; % alle Kräfte in [N]
end
% Plotten des Weges des Stempel 6
figure(1)
plot(Zeit1,Wege{1,6}); % Erstellen des Diagramms
xlabel('Zeit [s]') % Definieren der x-Achse
ylabel('Weg [mm]') % Definieren der y-Achse
title('Weg über die Zeit') % Definieren des Titels des Diagramms
% Plotten der Kraft des Stempel 6
figure(2)
plot(Zeit1,Kraefte{1,6}); % Erstellen des Diagramms
xlabel('Zeit [s]') % Definieren der x-Achse
ylabel('Kraft [N]') % Definieren der y-Achse
title('Kraft über die Zeit') % Definieren des Titels des Diagramms
%% Trennen der Daten an den Pausen
% Für jeden Weg (1-11) werden die Daten an den Aufnahmepausen getrennt
% Definieren der Zell-Arrays
Wege = evalin('base','Wege');
Zeit = evalin('base','Zeit1');
schwellenwert = 100;
datenVorSprungZeitW = cell(1,11);
datenVorSprungWeg = cell(1,11);
datenNachSprungZeitW = cell(1,11);
datenNachSprungWeg = cell(1,11);17.11.24 17:00 D:\Masterarbeit ...\BettungsansatzNeu.m 3 of 12
% Schleifendurchlauf führt zum trennen der Daten an den Speicherpausen
for i = 1:11
startIndex = 1; % Definieren des Start-Index für jeden Schleifendurchlauf
while startIndex < length(Zeit)
% Suche nach einem Sprung in den Zeitwerten (Unterschied > Schwellenwert)
% 'diff' berechnet die Differenz zwischen aufeinanderfolgenden Zeitwerten
% 'find' sucht ersten Index, an dem der Unterschied den Schwellenwert
überschreitet
% '+ startIndex -1' passten den Index relativ zum 'startIndex' an
sprungIndex = find(abs(diff(Zeit(startIndex:end)))>schwellenwert, 1) +
startIndex -1;
if isempty(sprungIndex)
% Speichert alle verbleibenden Daten nach dem letzten Sprung
datenNachSprungZeitW{i}{end+1} = Zeit(startIndex:end); % Zeit nach dem Sprung
speichern
datenNachSprungWeg{i}{end+1} = Wege{i}(startIndex:end); % Weg nach dem Sprung
speichern
break;
else
% Speichern der Daten vor dem Sprung
datenVorSprungZeitW{i}{end+1} = Zeit(startIndex:sprungIndex); % Zeit vor dem
Sprung speichern
datenVorSprungWeg{i}{end+1} = Wege{i}(startIndex:sprungIndex); % Weg vor dem
Sprung speichern
% Setzt den Start-Index auf den nächsten Wert
startIndex = sprungIndex + 1;
end
end
end
% Für jede Kraft (1-11) werden die Daten an den Aufnahmepausen getrennt
% Definieren der Zell-Arrays
Kraefte = evalin('base','Kraefte');
Zeit = evalin('base','Zeit1');
schwellenwert = 100;
datenVorSprungZeitK = cell(1,11);
datenVorSprungKraft = cell(1,11);
datenNachSprungZeitK = cell(1,11);
datenNachSprungKraft = cell(1,11);
% Schleifendurchlauf führt zum trennen der Daten an den Speicherpausen
for i = 1:11
startIndex = 1; % Definieren des Start-Index für jeden Schleifendurchlauf
while startIndex < length(Zeit)
% Suche nach einem Sprung in den Zeitwerten (Unterschied > Schwellenwert)
% 'diff' berechnet die Differenz zwischen aufeinanderfolgenden Zeitwerten
% 'find' sucht ersten Index, an dem der Unterschied den Schwellenwert17.11.24 17:00 D:\Masterarbeit ...\BettungsansatzNeu.m 4 of 12
überschreitet
% '+ startIndex -1' passten den Index relativ zum 'startIndex' an
sprungIndex = find(abs(diff(Zeit(startIndex:end)))>schwellenwert, 1) +
startIndex -1;
if isempty(sprungIndex)
% Speichert alle verbleibenden Daten nach dem letzten Sprung
datenNachSprungZeitK{i}{end+1} = Zeit(startIndex:end); % Zeit nach dem Sprung
speichern
datenNachSprungKraft{i}{end+1} = Kraefte{i}(startIndex:end); % Weg nach dem
Sprung speichern
break;
else
% Speichern der Daten vor dem Sprung
datenVorSprungZeitK{i}{end+1} = Zeit(startIndex:sprungIndex); % Zeit vor dem
Sprung speichern
datenVorSprungKraft{i}{end+1} = Kraefte{i}(startIndex:sprungIndex); % Weg vor
dem Sprung speichern
% Setzt den Start-Index auf den nächsten Wert
startIndex = sprungIndex + 1;
end
end
end
% Plotten des ersten Zeitabschnittes des Weges von Stempel 6
figure(3)
plot(datenVorSprungZeitW{1,6}{1,1},datenVorSprungWeg{1,6}{1,1}) % Erstellen des
Diagramms
xlabel('Zeit [s]') % Definieren der x-Achse
ylabel('Weg [mm]') % Definieren der y-Achse
title('Erster Zeitabschnitt des Weges (Stempel 6)') % Definieren des Titels des
Diagramms
% Plotten des ersten Zeitabschnittes der Kraft von Stempel 6
figure(4)
plot(datenVorSprungZeitK{1,6}{1,1},datenVorSprungKraft{1,6}{1,1}) % Erstellen des
Diagramms
xlabel('Zeit [s]') % Definieren der x-Achse
ylabel('Kraft [N]') % Definieren der y-Achse
title('Erster Zeitabschnitt der Kraft (Stempel 6)') % Definieren des Titels des
Diagramms
%% Bestimmen der Hoch- und Tiefpunkte sowie die Differenz dazwischen
% Erkennen der Hoch und Tiefpunkte der einzelnen Verformungen zu jeder Zeit
% Definieren der Zell-Arrays
TP_W6_300 = cell(1);
TP_W6_Zeit_300 = cell(1);
for j = 1:length(datenVorSprungWeg{1,6})17.11.24 17:00 D:\Masterarbeit ...\BettungsansatzNeu.m 5 of 12
% 'ylower' finden der Tiefpunkte des Weges bei Stempel 6 durch Envelope Kurve
[~,ylower] = envelope(datenVorSprungWeg{1,6}{j},300,'peak');
% alle Werte finden an denen die Envelope Kurve die Tiefpunkte schneidet
indicesLower = find(ylower == datenVorSprungWeg{1,6}{j});
% finden des 300ten Tiefpunkt und abspeichern des 300ten Tiefpunktes des Weges
von Stempel 6
TP_W6_300{j} = datenVorSprungWeg{1,6}{j}(indicesLower(300));
% abspeichern der entsprechenden Zeitwerte zu den Tiefpunkten
TP_W6_Zeit_300{j} = datenVorSprungZeitW{1,6}{j}(indicesLower(300));
end
Tiefpunkte_W_300 = cell(1,11); % Definieren des Zell-Arrays
TP_W_Zeit = cell(1,11);
for i = 1:11
for j = 1:length(TP_W6_Zeit_300)
% Finde den Index, an dem die Zeitpunkte des 300ten Teifpunktes mit den
Zeitpunkten in 'datenVorSprungZeitW{i}{j}' übereinstimmen
IndexZeitenTP = find(TP_W6_Zeit_300{j} == datenVorSprungZeitW{i}{j});
% Speichern der Werte der Wege an der Position des 'IndexZeitenTP' im Zell-
Array 'Tiefpunkte_W_300'
Tiefpunkte_W_300{i}{j} = datenVorSprungWeg{i}{j}(IndexZeitenTP);
end
end
% Definieren der Zell-Arrays
Hochpunkt_W_300 = cell(1,11);
HP_W_Zeit_300 = cell(1,11);
for i = 1:11
for j = 1:length(TP_W6_Zeit_300)
% verschieben der Zeit um 0,05s zu der Zeit des 300ten Tiefpunktes
verschobene_ZeitW = TP_W6_Zeit_300{j} + 0.05;
% finden der verschobenen Zeit in den Zeiten von Stempel 6
[~, index] = min(abs(datenVorSprungZeitW{1,6}{j} - verschobene_ZeitW));
% speichern der Hochpunkte der Wege zu der verschobenen Zeit
Hochpunkt_W_300{i}{j} = datenVorSprungWeg{i}{j}(index);
% speichern der zugehörigen Zeit zu den Hochpunkten
HP_W_Zeit_300{i}{j} = datenVorSprungZeitW{i}{j}(index);
end
end
% Berechnung der Differenzen zwischen dem 300ten Hoch- und Tiefpunkten jedes Weges
WegDiff = zeros(11,1); % Definieren der Matrix
for i = 1:11
for j = 1:length(TP_W6_Zeit_300)
% Berechnung der Differenz
WegDiff(i,j) = Tiefpunkte_W_300{i}{j} - Hochpunkt_W_300{i}{j};
end
end17.11.24 17:00 D:\Masterarbeit ...\BettungsansatzNeu.m 6 of 12
% Erkennen der Hoch und Tiefpunkte der einzelnen Kraft zu jeder Zeit
% Definieren der Zell-Arrays
TP_K6_300 = cell(1);
TP_K6_Zeit_300 = cell(1);
for j = 1:length(datenVorSprungKraft{1,6})
% 'ylower' finden der Tiefpunkte der Kraft bei Stempel 6 durch Envelope Kurve
[~,ylower] = envelope(datenVorSprungKraft{1,6}{j},350,'peak');
% alle Werte finden an denen die Envelope Kurve die Tiefpunkte schneidet
indicesLowerK = find(ylower == datenVorSprungKraft{1,6}{j});
% finden des 300ten Tiefpunkt und abspeichern des 300ten Tiefpunktes der Kraft
von Stempel 6
TP_K6_300{j} = datenVorSprungKraft{1,6}{j}(indicesLowerK(300));
% abspeichern der entsprechenden Zeitwerte zu den Tiefpunkten
TP_K6_Zeit_300{j} = datenVorSprungZeitK{1,6}{j}(indicesLowerK(300));
end
Tiefpunkte_K_300 = cell(1,11); % Definieren des Zell-Arrays
for i = 1:11
for j = 1:length(TP_K6_Zeit_300)
% Finde den Index, an dem die Zeitpunkte des 300ten Teifpunktes mit den
Zeitpunkten in 'datenVorSprungZeitK{i}{j}' übereinstimmen
IndexZeitenTP = find(TP_K6_Zeit_300{j} == datenVorSprungZeitK{i}{j});
% Speichern der Werte der Wege an der Position des 'IndexZeitenTP' im Zell-
Array 'Tiefpunkte_K_300'
Tiefpunkte_K_300{i}{j} = datenVorSprungKraft{i}{j}(IndexZeitenTP);
end
end
Hochpunkt_K_300 = cell(1,11);
HP_K_Zeit_300 = cell(1,11);
for i = 1:11
for j = 1:length(TP_K6_Zeit_300)
% verschieben der Zeit um 0,05s zu der Zeit des 300ten Tiefpunktes
verschobene_ZeitK = TP_K6_Zeit_300{j} + 0.05;
% finden der verschobenen Zeit in den Zeiten von Stempel 6
[~, index] = min(abs(datenVorSprungZeitK{1,6}{j} - verschobene_ZeitK));
% speichern der Hochpunkte der Kraft zu der verschobenen Zeit
Hochpunkt_K_300{i}{j} = datenVorSprungKraft{i}{j}(index);
% speichern der zugehörigen Zeit zu den Hochpunkten
HP_K_Zeit_300{i}{j} = datenVorSprungZeitK{i}{j}(index);
end
end
% Berechnung der Differenzen zwischen dem 300ten Hoch- und Tiefpunkten jeder Kraft
KraftDiff = zeros(11,1); % Definieren der Matrix
for i = 1:11
for j =1:length(TP_K6_Zeit_300)
% Berechnung der Differenz
KraftDiff(i,j) = Tiefpunkte_K_300{i}{j} - Hochpunkt_K_300{i}{j};17.11.24 17:00 D:\Masterarbeit ...\BettungsansatzNeu.m 7 of 12
end
end
%% Regression des Weges
fitresults = cell(1, size(WegDiff,2)); % Definieren eines Zell-Array für die Fit-
Ergebnisse
gofs = cell(1, size(WegDiff,2)); % Definieren eines Zell-Array für die Goodness-of-
Fit-Daten
ft = fittype( 'gauss1' ); % Definieren des Fit-Typs als Gaußsche Funktion
% Festlegen der Fit-Optionen, hier wird nichtlineare kleinste Quadrate verwendet
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
% Deaktivieren der Ausgabe von Informationen während der Anpassung
opts.Display = 'Off';
% Festlegen der unteren Schranken für die Parameter des Fits
opts.Lower = [-Inf -Inf 0];
for i = 1:size(WegDiff,2) % Schleife über alle Spalten von 'WegDiff'
yData = WegDiff(:,i); % Extrahiere die i-te Spalte von 'WegDiff' als y-Daten
xData = LS; % Verwendet die Position der Laststempel als x-Daten
% Bereitet die Daten für die Kurvenanpassung vor
[xData, yData] = prepareCurveData( xData, yData );
% Startwerte für die Parameter des Gauss-Fits, [Amplitude, Lage, Streuung]
opts.StartPoint = [-0.0556957572698593 250 150];
% Durchführung des Fits mit den vorbereiteten x- und y-Daten, dem Fit-Typ und den
Optionen
[fitresult, gof] = fit( xData, yData, ft, opts );
% Speichern des Fit-Ergebnisses und der Güte der Anpassung
fitresults{i} = fitresult;
gofs{i} = gof;
% Erstellen einer neuen Figur nur alle 10 Datenpunkte oder für die letzte Spalte
if mod(i, 10) == 1 || i == size(WegDiff, 2)
% Der Name des Diagramms enthält den aktuellen Index 'i'
figure('Name', ['Gauss Regression Weg Laststempel ', num2str(i)]);
% Plotten des Fits zusammen mit den Daten
h = plot(fitresult, xData, yData);
% Hinzufügen einer Legende zur Grafik
legend(h, ['Weg', num2str(i)], ['Gauss Regression Weg', num2str(i)],
'Location', 'NorthEast', 'Interpreter', 'none');
% Beschriften der x-Achse
xlabel('Laststempel [mm]', 'Interpreter', 'none');
% Beschriften der y-Achse
ylabel('Weg [mm]', 'Interpreter', 'none');
% Gitternetz in Grafik einblenden
grid on;
% Einfügen des R²-Wertes als Text in die Grafik
text(mean(xData)/2, mean(yData)/2, ['R^2 = ', num2str(gof.rsquare)],
'FontSize', 14);17.11.24 17:00 D:\Masterarbeit ...\BettungsansatzNeu.m 8 of 12
end
end
% Definieren der Matrizen
AW = zeros(1,length(WegDiff));
BW = zeros(1,length(WegDiff));
CW = zeros(1,length(WegDiff));
for i=1:length(WegDiff)
% Speichern des Regressionsparameter a
AW(1,i) = fitresults{1,i}.a1;
% Speichern des Regressionsparameter b
BW(1,i) = fitresults{1,i}.b1;
% Speichern des Regressionsparameter c
CW(1,i) = fitresults{1,i}.c1;
end
%% Erstellen der Funktion des Weges und berechnen der 4ten Ableitung
% Definieren von Matrizen
fW_Funktion = cell(1, size(WegDiff, 2)); % Speichern der Gauß-Funktionen
fW4_Funktion = cell(1, size(WegDiff, 2)); % Speichern der vierten Ableitungen
fW= zeros(1, size(WegDiff, 2)); % Wert der Funktion bei x = 250
fW4 = zeros(1, size(WegDiff, 2)); % Wert der vierten Ableitung bei x = 250
for i = 1:size(WegDiff, 2)
% Extrahieren der Regressionsparameter für die i-te Funktion
a = AW(1, i); % Amplitude
b = BW(1, i); % Lage
c = CW(1, i); % Streuung
% Definieren der Gauß-Funktion fw als anonyme Funktion
fW_Funktion{i} = @(x) a * exp(-((x - b).^2) / (c^2));
% Berechnen der vierten Ableitung der Funktion fw
syms x_1; % Symbolische Variable für Ableitungen
fW_sym = a * exp(-((x_1 - b)^2) / (c^2)); % Symbolische Form der Funktion
fW4_sym = diff(fW_sym, x_1, 4); % Vierte Ableitung
% Umwandeln in eine anonyme Funktion für die vierte Ableitung
fW4_Funktion{i} = matlabFunction(fW4_sym);
% Auswerten der Funktion und der vierten Ableitung bei x = 250
fW(i) = fW_Funktion{i}(250); % Wert der Funktion bei x = 250
fW4(i) = fW4_Funktion{i}(250); % Wert der vierten Ableitung bei x = 250
end
%% Regression der Kraftdaten
fitresults = cell(1, size(KraftDiff,2)); % Definieren eines Zell-Array für die Fit-
Ergebnisse
gofs = cell(1, size(KraftDiff,2)); % Definieren eines Zell-Array für die Goodness-of-17.11.24 17:00 D:\Masterarbeit ...\BettungsansatzNeu.m 9 of 12
Fit-Daten
ft = fittype( 'gauss1' ); % Definieren des Fit-Typs als Gaußsche Funktion
% Festlegen der Fit-Optionen, hier wird nichtlineare kleinste Quadrate verwendet
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
% Deaktivieren der Ausgabe von Informationen während der Anpassung
opts.Display = 'Off';
% Festlegen der unteren Schranken für die Parameter des Fits
opts.Lower = [-Inf -Inf 0];
for i = 1:size(KraftDiff,2) % Schleife über alle Spalten von 'KraftDiff'
yData = KraftDiff(:,i); % Extrahiere die i-te Spalte von 'KraftDiff' als y-Daten
xData = LS; % Verwendet die Position der Laststempel als x-Daten
% Bereitet die Daten für die Kurvenanpassung vor
[xData, yData] = prepareCurveData( xData, yData );
% Startwerte für die Parameter des Gauss-Fits, [Amplitude, Lage, Streuung]
opts.StartPoint = [73.9783163070679 130 79.5354684867071];
% Durchführung des Fits mit den vorbereiteten x- und y-Daten, dem Fit-Typ und den
Optionen
[fitresult, gof] = fit( xData, yData, ft, opts );
% Speichern des Fit-Ergebnisses und der Güte der Anpassung
fitresults{i} = fitresult;
gofs{i} = gof;
% Erstellen einer neuen Figur nur alle 10 Datenpunkte oder für die letzte Spalte
% Erstellen einer neuen Figur für die Visualisierung des Fits
if mod(i, 10) == 1 || i == size(KraftDiff, 2)
% Der Name des Diagramms enthält den aktuellen Index 'i'
figure( 'Name', ['Gauss Regression Kraft Laststempel', num2str(i)] );
% Plotten des Fits zusammen mit den Daten
h = plot( fitresult, xData, yData );
% Hinzufügen einer Legende zur Grafik
legend( h, ['Kraft', num2str(i)], ['Gauss Regression Kraft', num2str(i)],
'Location', 'NorthEast', 'Interpreter', 'none' );
% Beschriften der x-Achse
xlabel( 'Laststempel [mm]', 'Interpreter', 'none' );
% Beschriften der y-Achse mit der entsprechenden Kraft-Spalte
ylabel( 'Kraft [N]', 'Interpreter', 'none' );
% Gitternetz in Grafik einblenden
grid on
% Einfügen des R²-Wertes als Text in die Grafik
text(mean(xData)/2, mean(yData)/2, ['R^2 = ', num2str(gof.rsquare)],
'FontSize',14)
end
end
% Definieren der Matrizen
AK = zeros(1,length(KraftDiff));
BK = zeros(1,length(KraftDiff));17.11.24 17:00 D:\Masterarbeit ...\BettungsansatzNeu.m 10 of 12
CK = zeros(1,length(KraftDiff));
for i=1:length(KraftDiff)
% Speichern des Regressionsparameter a
AK(1,i) = fitresults{1,i}.a1;
% Speichern des Regressionsparameter b
BK(1,i) = fitresults{1,i}.b1;
% Speichern des Regressionsparameter c
CK(1,i) = fitresults{1,i}.c1;
end
%% Erstellen der Funktion der Kraft
% Definieren der Matrizen
fK_Funktion = cell(1, size(KraftDiff, 2)); % Speichern der Gauß-Funktionen
fK = zeros(1, size(KraftDiff, 2)); % Wert der Funktion bei x = 250
for i = 1:size(KraftDiff, 2)
% Extrahieren der Regressionsparameter für die i-te Funktion
a = AK(1, i); % Amplitude
b = BK(1, i); % Lage
c = CK(1, i); % Streuung
% Definieren der Gauß-Funktion fk als anonyme Funktion
fK_Funktion{i} = @(x) a * exp(-((x - b).^2) / (c^2));
% Auswerten der Funktion bei x = 250
fK(i) = fK_Funktion{i}(250); % Wert der Funktion bei x = 250
end
%% Berechenen des Elastizitätsmoduls über den Hetenyi Bettungsansatz
D = zeros(1,length(fW4)); % Definieren einer Matrix
I = (500*150^3)/12; % Flächenträgheitsmoment eines Rechteck in [mm^4]
E = zeros(1,length(fW4)); % Definieren einer Matrix
for i= 1:length(fW4)
% Berechnung der Plattensteifigkeit und das Elastizitätsmodul
D(i) = double((-35200 * fW(i) + fK(i))/(fW4(i)));
E(i) = abs(double(D(i)*(1-0.35^2))/I);
end
%% Berechnen des ersten Lastwechsel pro Zeitstufe (lange Versuche)
% Spalten = size(E, 2); % Anzahl der Spalten basierend auf Matrix E
% Lastzyklen = 600 + (0:Spalten-1) * 9000; % Füllen der Matrix mit den Lastzyklen pro
Zeitstufe
%% Berechnen des ersten Lastwechsel pro Zeitstufe (kurze Versuche)
Spalten = size(E, 2); % Anzahl der Spalten basierend auf Matrix Es
Lastzyklen = 200 + (0:Spalten-1) * 3000; % Füllen der Matrix mit den Lastzyklen pro
Zeitstufe17.11.24 17:00 D:\Masterarbeit ...\BettungsansatzNeu.m 11 of 12
%% Berechnung der Energy Ratio für den Versuch
EnergyRatio = Lastzyklen .* E; % Elementweise Multiplikation der Vektoren
figure
plot(Lastzyklen, EnergyRatio, 'o'); % Erstellen des Diagramms der Energy Ratio
xlabel('Lastzyklen [-]') % Definieren der x-Achse
ylabel('ERn [-]') % Definieren der y-Achse
title('Energy Ratio') % Definieren des Titels des Diagramms
%% Berechenen der Werte für die Regression
[Max, Zeitstufe] = max(EnergyRatio); % Maximalwert und passende Zeitstufe der Energy
Ratio
% Berechnen der Indizes für ±20% rund um die Zeitstufe des Max-Wertes
minus20 = max(round(Zeitstufe * 0.8), 1); % Mindestens 1
plus20 = min(round(Zeitstufe * 1.2), length(EnergyRatio)); % Maximal die Länge von
EnergyRatio
WerteUmMax = EnergyRatio(minus20:plus20); % Finde die Werte in Energy Ratio zwischen
-20% und +20%
Lastzyklen20 = Lastzyklen(minus20:plus20); % Speichere die passenden Lastzyklen in
einer Matrix
%% Fit: 'Regression 4ten Grades der Energy Ratio'.
[xData, yData] = prepareCurveData( Lastzyklen20, WerteUmMax );
% Set up fittype and options.
ft = fittype( 'poly4' );
% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft );
% Plot fit with data.
figure( 'Name', 'Regression 4ten Grades der Energy Ratio' );
h = plot( fitresult, xData, yData );
legend( h, 'WerteUmMax vs. Lastzyklen20', 'Regression 4ten Grades der Energy Ratio',
'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'Lastzyklen [-]', 'Interpreter', 'none' );
ylabel( 'ERn [N/mm^2]', 'Interpreter', 'none' );
grid on
% Abrufen der Regressionsparameter
coefficients = coeffvalues(fitresult);
% Speichere die Regressionsparameter in separate Vektoren
w4 = coefficients(1);
w3 = coefficients(2);
w2 = coefficients(3);
w1 = coefficients(4);
w0 = coefficients(5);17.11.24 17:00 D:\Masterarbeit ...\BettungsansatzNeu.m 12 of 12
%% Erstellen der Regerssionsfunktion und finden den Hochpunkt
syms x3 % Definiert eine symbolische Variable 'x3'
fNMakro = symfun(w4*x3^4 + w3*x3^3 + w2*x3^2 + w1*x3 + w0, x3); %Regressionsfunktion
% Berechnen der 1. Ableitung von fNMakro nach 'x3' über diff
fNMakro1 = diff(fNMakro, x3);
% Nullstellen berechnen (Notwendige Bedingung)
Nullstellen = solve(fNMakro1 == 0, x3);
% Berechne die zweite Ableitung (Hinreichende Bedingung)
fNMakro2 = diff(fNMakro1, x3);
% Finde den Hochpunkt durch Überprüfung der zweiten Ableitung
for i = 1:length(Nullstellen)
% Setze die Nullstellen in die zweite Ableitung ein
zweiteAbleitungWert = double(subs(fNMakro2, x3, Nullstellen(i)));
% Prüfe, ob die Nullstelle ein Hochpunkt ist (zweite Ableitung < 0)
if zweiteAbleitungWert < 0
NMakroX = double(Nullstellen(i)); %Definieren des x-Wertes von NMakro
NMakroY = double(subs(fNMakro, x3, NMakroX)); %Definieren des y-Wertes von
NMakro
break;
end
end
% Ausgabe des Hochpunkts
disp(['Hochpunkt der Regressionsfunktion (LW, ERn): (', num2str(NMakroX), ', ',
num2str(NMakroY), ')']);
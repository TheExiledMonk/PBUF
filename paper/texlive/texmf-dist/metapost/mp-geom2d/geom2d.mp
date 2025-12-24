%
if not known mplib: input latexmp fi;

% Don't load this package twice:
if known mpgeomdd_version: expandafter endinput; fi;

numeric mpgeomdd_version;string mpgeomdd_date;
mpgeomdd_version=1.4;
mpgeomdd_date="2025/12/03";
% The banner:
message "******* mp-geom2d " & decimal (mpgeomdd_version) &
        " (c) Jean-Michel Sarlat, Maxime Chupin (" & mpgeomdd_date & ") *******";message "";

input geom2d-arc;
input geom2d-c2d;
input geom2d-fct;
input geom2d-lbl;
input geom2d-plt;
input geom2d-rep;
input geom2d-tra;
input geom2d-svgnames;
input geom2d-main;
input graph;

if known loadHershey: expandafter endinput fi

% this default is for Debian with ‘hershey-fonts-data’ installed.
newinternal string hersheyFonts;
hersheyFonts := "/usr/share/hershey-fonts/";

vardef loadHershey @# =
    if known hershey.loaded.@# :
        message("Hershey font "&str(@#)&" already known: skipping rereading.");
    else:
        hershey.loaded.font := 1;
        doLoadHershey.@# ;
    fi
enddef;
vardef doLoadHershey suffix font =
    % create the variable hershey.<fontname>
    save var; string var; var := "hershey." & str font & ".";
    scantokens ("path "&var)[][];
    % variables used for reading the file
    save   file, line;
    string file, line;
    file := hersheyFonts & str font & ".jhf";
    save pnum, lnum, wd, i, x, y, first;
    boolean first; lnum := 0;
    % read the font line by line
    forever :
        line := readfrom file;
        exitif line = EOF;
        lnum := lnum + 1; pnum := 1; first := true;
        % the line starts with a width, followed by min/max x coordinates, 
        % followed by the data describing the glyph, in pairs of coordinates,
        % every coördinate encoded as a single character.
        wd   := scantokens (substring(5,8) of line) - 1;
        if wd > 0 :
            line := substring (8, infinity) of line;
            x := ASCII(line) - 82; line := substring(1, infinity) of line;
            y := ASCII(line) - 82; line := substring(1, infinity) of line;
            scantokens(var)[lnum].lft := x;
            scantokens(var)[lnum].rt := y;
            %message char(lnum+31) & " (" & decimal(lnum) & "): (" & decimal x & ", " & decimal y & ")";
            scantokens(var)[lnum][pnum] := fi
        for i = 1 upto wd :
            hide( % 82 is ASCII("R")
                x := ASCII(line) - 82; line := substring(1, infinity) of line;
                y := 82 - ASCII(line); line := substring(1, infinity) of line;)
            if (x,y) = (-50,0) :
                % the special string " R" separates paths
                ; pnum := pnum+1; first := true;
                scantokens(var)[lnum][pnum] :=
            else :
                if first : hide(first := false)
                else : -- fi (x,y)
            fi
        endfor;
    endfor
    closefrom file;
enddef;

vardef drawHershey @# (expr s) text t =
    if numeric s: drawHersheyChar @# (s) t;
    else : drawHersheyString @# (s) t; fi
enddef;
vardef drawHersheyChar @# (expr c) text t =
    save i; i := 1; forever :
        exitif unknown hershey.@#[c][i];
        draw hershey.@#[c][i] t;
        i := i + 1;
    endfor
enddef;
vardef drawHersheyString @# (expr s) text t =
    save c, ss, sh; string ss; ss := s; sh := 0;
    forever :
        c := ASCII(ss) - 31;
        sh := sh + hershey.@#[c].rt;
        % NB. -9 is the baseline of the hershey fonts;
        % we put the baseline on x=0 instead.
        drawHersheyChar @# (c) shifted (sh, 9) t;
        sh := sh - hershey.@#[c].lft;
        ss := substring(1, infinity) of ss;
        exitif length(ss) = 0;
    endfor
enddef;


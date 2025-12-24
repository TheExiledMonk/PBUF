
string lblS[];
lblS[0] = "alpha";
lblS[1] = "beta";
lblS[2] = "gamma";
lblS[3] = "delta";
lblN    = 3;


vardef lTEX primary s = 
  textext("$"&scanchaine_label(s)&"$")
enddef;



vardef gddLabel.@#(expr s,p) text t_=
  % comme label mais qui s’applique à un point
  % s : matériel de l’annotation 
  % p : point mpgdd
  label.@#(s,if(numeric p):Pt(p)else:p fi gddEnPlace) t_
enddef;

vardef Etiquette.@#(expr s,t,p) = label.@#(textext(s) scaled t,p gddEnPlace)
enddef;

vardef Tag.@#(expr s,t,p) = Etiquette.@#(s,t,p) enddef;

vardef EtiquetteChemin.@#(expr s,p,pos) = % label, chemin, position
  save chemin;
  path chemin;
  if path p:
    chemin:=p;
  else:
    chemin:=gddP[p]; % fonction pour chemin et courbe
  fi;
  label.@#(textext(s), _milieuChemin(chemin,pos) gddEnPlace)
enddef;

vardef PathTag.@#(expr s,p,pos) = EtiquetteChemin.@#(s,p,pos) enddef;

vardef _milieuChemin(expr p,pos) =  % p est du type path
  save midtime;
  midtime = arctime (pos*(arclength(p))) of p;
  point infinity of (subpath (0,midtime) of p)
enddef;

vardef FICHIER primary s =
    save _s; string _s; _s = "input " & s;
    scantokens _s
enddef;

vardef m_arque.@# expr p =
    pointe p;
    label.@#(gddL[p] scaled gddTailleLabel,PointTOPair(p) gddEnPlace);
enddef;

vardef m_a_rque.@# expr p =
    pointe(scantokens p);
    label.@#(lTEX(p) scaled gddTailleLabel,PointTOPair(scantokens p) gddEnPlace);
enddef;

vardef marque.@# expr p =
   if string p: m_a_rque.@# p else: m_arque.@# p fi
enddef;


vardef gddMark.@# expr p = marque.@# p enddef;

vardef scanchaine_label(expr s) =
  save d,m,f,c,l,flag,i; string d,m,f,c;
  d := ""; m := ""; f := "";
  l = length(s); flag := 0;
  for i:=0 upto l:
    c := substring (i,i+1) of s;
    if c = "'":
      f := f & c; flag := 2;
    elseif c = "_":
      flag := 1;
    else:
      if flag = 0:
        d := d & c
      else:
        m := m & c
      fi;
    fi
  endfor;
  for i:=0 upto lblN:
    if d = lblS[i]: d := "\" & d fi
  endfor;
  d := d & "_{" & m & "}" & f; d
enddef;

endinput
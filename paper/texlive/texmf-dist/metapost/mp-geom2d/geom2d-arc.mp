%%% --------------------------------------------------------------------------
%%% Arcs de cercles
%%% --------------------------------------------------------------------------

vardef Arc(expr c,r,a,b) =
  % c: centre de l’arc (point ou pair)
  % r: rayon (numeric)
  % a: angle de départ (numeric)
  % b: angle d’arrivée (numeric)
  save n,centre,pas; n = incr gddO;
  gddT[n] = "chemin";
  pair centre; centre = PairImp(c); pas =(b-a)/10;
  gddP[n]:=(centre+r*(cos(a),sin(a))) for i=0 upto 10: ..
  (centre+r*(cos(a+i*pas),sin(a+i*pas))) endfor;
  n
enddef;

vardef ArcEntrePoints(expr C, r, A, B,s) =
  % C : centre de l’arc (point ou pair)
  % r : rayon (numeric)
  % A : point ou pair de départ de l’arc
  % B : point ou pair d’arrivée
  % s : sens 1 ou -1
  save n,cercle,t; n = incr gddO;
  gddT[n] = "chemin";
  numeric t[];
  path cercle;
  cercle := fullcircle scaled 2r shifted PairImp(C);
  (t1,t2) = cercle intersectiontimes (PairImp(C)--5[PairImp(C),PairImp(A)]);
  (t3,t4) = cercle intersectiontimes (PairImp(C)--5[PairImp(C),PairImp(B)]);
  
  if(t1<t3):
    t5 = t1;
    t6 = t3;
  else:
    t5 = t3;
    t6 = t1;
  fi
  if(s=1):
    gddP[n] := subpath (t5,t6) of cercle;
  else:
    gddP[n] := subpath (t5,t6-8) of cercle;
  fi
  n 
enddef;

vardef ArcBetweenPoints(expr C, r, A, B,s) = ArcEntrePoints(C,r,A,B,s) enddef;;


%   obtenir seulement une partie d’un cercle déjà défini
vardef gddTraceArcDeCercle(expr C,r,d)= 
  % C : cercle 4
  % r : premier angle en radian
  % d : deuxième angle en radian
  (subpath(r/Pi*4,d/Pi*4) of gddTraceObjet(C))
enddef;

vardef gddDrawCircleArc(expr C,r,d) = gddTraceArcDeCercle(C,r,d) enddef;

endinput
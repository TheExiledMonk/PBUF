%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                        mptrees.mp                          %%
%%               Probability trees with MetaPost              %%
%%                    o.peault@posteo.net                     %%
%%                  Version 24.04 (April 2024)                %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
string mptreesversion;
mptreesversion:="24.04";
%
% This work may be distributed and/or modified under the conditions of
% the LaTeX Project Public License, either version 1.3 of this license
% or (at your option) any later version.  The latest version of this
% license is in http://www.latex-project.org/lppl.txt
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

message "mptrees version " & mptreesversion;

if not known mplib: input latexmp fi;

boolean arbres_latexmp;
arbres_latexmp:=true;


boolean mpt_begtr; % is begintree used ?
mpt_begtr:=false;
numeric mpt_passe;
mpt_passe:=1;


% begintree permet de stocker les instructions de la figure dans mpt_tree pour calculs éventuels

def begintree=
 mpt_begtr:=true;
 scantokens "def mpt_tree="
enddef;

let endtree = enddef;

numeric mpt_tmpdirtree;


% à la fin de begintree, on exécute deux fois mpt_tree
extra_endfig:="if mpt_begtr: mpt_tree currentpicture:=nullpicture; mpt_passe:=2; AdjustNodes; mpt_tree; mpt_begtr:=false;mpt_passe:=1 fi;" & extra_endfig;


numeric posprob;   % position de l'étiquette sur la branche
posprob:=-1;

numeric scaleprob;
scaleprob:=0.85;   % échelle de l'étiquette sur la branche

numeric scaleev;
scaleev:=1;   % échelle de l'évènement

numeric dirtree,dirlabel; % direction de l'arbre
dirtree:=0;
dirlabel:=0;

numeric proboffset;  % redéfinition temporaire de labeloffset pour affichage probas
proboffset:=labeloffset;

numeric endedgeshift;
endedgeshift:=0;

numeric brokenlineratio;
brokenlineratio:=0.2;

numeric linewidth;
linewidth:=0.5bp;

color linecolor;
linecolor:=black;

numeric nodelinewidth; % épaisseur de la ligne autour des noeuds
nodelinewidth:=-1; % change pour tree et graph

pair Orig_arbre[][];  % sommet de l'arbre i,j
Orig_arbre[1][1]:=origin;

pair mpt_dec_ori[][]; % Décalage avec la fin de la branche précédente
mpt_dec_ori[1][1]:=(0,0);

pair mpt_tmp_ori[][];  % Branche i,j,k

pair mpt_br[][][];  % Branche i,j,k

boolean exist.arbre[][]; % L'arbre i,j est-il dessiné ?
exist.arbre[0][1]:=true;


numeric colonne_cour,ligne_cour;
numeric mptLargeur[][],mptHauteur[][];  % diff largeur au rang i, diff hauteur rang i

numeric countligne[];  % nombre de ligne pour le niveau i
for i=0 upto 128: countligne[i]:=0; endfor

%numeric typedec;  % 0 origine dépend de la taille de l'évènement, 1 taille fixée
numeric shiftev;   % taille du décalage dans le cas où typedec=1
shiftev:=-1; % dans ce cas, typedec =0
%extra_beginfig:=extra_beginfig & "shiftev:=-1;"; % remise à zéro en début de figure


numeric typeprob; % affichage de la proba sur la branche
typeprob:=1;

boolean abscoord; % Coordonnée absolues
abscoord=false;

string branchtype,endlabeltype;
branchtype="segment";
endlabeltype="none";

boolean edgearrow;
edgearrow:=false;

numeric endlabelspace;
endlabelspace:=1cm;


% Pour rendre les paramètres "locaux" à chaque figure
extra_beginfig:=extra_beginfig & "init_loc_var;";



% Étiquette au début de l'arbre

picture mptreestartlabel;
mptreestartlabel:=nullpicture;

vardef startlabel(expr s)=
  colonne_cour:=0;
  mptreestartlabel:=labelarbres(s);
  if dirtree<>0: dirlabel:=dirtree fi;
   if dirlabel>0:
     theevlabel(formatnodeev(labelarbres(s)) scaled scaleev,origin,-180+dirlabel)
   else:
     theevlabel(formatnodeev(labelarbres(s)) scaled scaleev,origin,180+dirlabel)
   fi
enddef;


% Étiquettes de fin d'arbre
vardef colonne.endlabel[]=
  @
enddef;

vardef endlabel[][](expr s)=
  save i,j,tmpbranchtype,tmpabscoord,tmps;
  numeric i,j;
  string tmpbranchtype,tmps;
  boolean tmpabscoord;
  i:=colonne.#@;
  j:=@;
  if string s: tmps:= "\strut \phantom{.}" & s else: picture tmps;tmps:=s fi;
  tmpbranchtype:=branchtype;
  branchtype:=endlabeltype;
  tmpabscoord:=abscoord;
  abscoord:=false;
  tree[i][j]((endlabelspace,0) rotated (dirlabel-dirtree))(tmps," ")
  hide(branchtype:=tmpbranchtype;abscoord:=tmpabscoord;)
enddef;


vardef labelarbres(expr s)=
  save p; picture p;
  if picture s:  p=s
  elseif path s: p=image(draw s)
  else: p=textext(s)
  fi;
  p
enddef;


% longueur du segment qui traverse la figure p selon l'angle inc
vardef longdir(expr p,inc)=
  if urcorner p = llcorner p:
    0
  elseif (abs inc <= abs angle (urcorner p -llcorner p)) or (abs inc > 180 - abs angle (urcorner p -llcorner p)):
    abs((lrcorner p - llcorner p)/cosd(inc))
  else:
    abs((ulcorner p - llcorner p)/sind(inc))
  fi
enddef;

%%%%%%%%%%%%%%
%% format nodes, leaves and probabilities
%%%%%%%%%%%%%%

% Colors
color nodelinecolor,nodebgcolor,nodefgcolor;
nodelinecolor:=black;
nodebgcolor:=white;
nodefgcolor:=black;

color leavelinecolor,leavebgcolor,leavefgcolor;
leavelinecolor:=black;
leavebgcolor:=white;
leavefgcolor:=black;

color problinecolor,probbgcolor,probfgcolor;
problinecolor:=black;
probbgcolor:=white;
probfgcolor:=black;


% Superellipse
numeric superellipseparam;
superellipseparam:=0.85;

vardef superellipsebox(expr p)=
   save a,b,c,d,l;
   pair a,b,c,d;
   path l;
   l := bbox p;
   a=0.5[lrcorner l,urcorner l];
   b=0.5[urcorner l,ulcorner l];
   c=0.5[ulcorner l,llcorner l];
   d=0.5[llcorner l,lrcorner l];
   if (mpt_isgraph and (not naturalwidth) and (not mpt_isprob)):
      l:=superellipse(1/2 right, 1/2 up, 1/2 left, 1/2 down, superellipseparam);
      l:=l scaled nodewidth
   elseif naturalwidth:
      l:=superellipse(a,b,c,d,superellipseparam);
      l:=l shifted -center l
   else:
      l:=superellipse(a,b,c,d,superellipseparam)
   fi;
   l
enddef;


% Circle
vardef circlebox(expr p)=
  save a,b,c;
  pair a,b,c;
  a:=urcorner bbox p;
  b:=llcorner bbox p;
  c:=0.5[a,b];
  if (mpt_isgraph and (not naturalwidth) and (not mpt_isprob)):
     fullcircle scaled nodewidth
  elseif naturalwidth and (not mpt_isprob):
     fullcircle scaled (abs(b-a)-2)
  else:
     fullcircle scaled (abs(b-a)-2) shifted c
  fi
enddef;


% square
vardef squarebox(expr p)=
  save H,L;
  numeric H,L;
  L=abs((urcorner bbox p) - (ulcorner bbox p));
  H=abs((urcorner bbox p) - (lrcorner bbox p));
  if H>L: L:= H fi;
  if (mpt_isgraph and (not naturalwidth) and (not mpt_isprob)):
    unitsquare shifted (-0.5,-0.5) scaled nodewidth
  elseif naturalwidth:
     unitsquare shifted (-0.5,-0.5) scaled L
  else:
     unitsquare shifted (-0.5,-0.5) scaled L shifted center bbox p
  fi
enddef;






vardef circleprobformat(expr p,cf)=
image(%
 myfill circlebox(p) withcolor background;
 draw p withcolor cf _op_ mpt_po_;
)
enddef;





vardef formatprobbox(expr p,cl,cb,cf)=
image(%
 if (cb<>white) or (mpt_isprobover) or (mpt_bgc): myfill probshape(p) withcolor cb mpt_opbg_ fi;
 draw probshape(p) withcolor cl withpen pencircle scaled nodelinewidth mpt_po_;
 if arbrebool or mpt_isprob: draw p withcolor cf mpt_po_ fi;
)
enddef;

vardef formatnodebox(expr p,cl,cb,cf)=
save pp;path pp;
if mpt_isgraph: pp:= mpt_nodepath[num] shifted -mpt_nodecoord[num] else: pp:=nodeshape(p) fi;
image(%
 if (cb<>white) or (mpt_isprobover) or (mpt_bgc): fill pp withcolor cb mpt_opbg_ fi;
 draw pp withcolor cl withpen pencircle scaled nodelinewidth mpt_po_;
 if arbrebool or mpt_isprob: draw p shifted (center pp-center p) withcolor cf mpt_po_ fi;
)
enddef;

vardef formatleavebox(expr p,cl,cb,cf)=
image(%
 if (cb<>white) or (mpt_isprobover) or (mpt_bgc): fill leaveshape(p) withcolor cb mpt_opbg_ fi;
 draw leaveshape(p) withcolor cl withpen pencircle scaled nodelinewidth mpt_po_;
 if arbrebool or mpt_isprob: draw p withcolor cf mpt_po_ fi;
)
enddef;


% Format nodes

string nodeformat,leaveformat,probformat;  %How nodes and leaves should be printed
nodeformat:="";
leaveformat:="";
probformat:="";



vardef formatprob(expr p)=
save pp;path pp;
pp:=probshape(p);
 save cl,cb,cf;
 color cl,cb,cf;
 cl:=problinecolor;
 cb:=probbgcolor;
 cf:=probfgcolor;
 if mpt_isprobover and (probformat="") and mpt_probsh_ori:
    circleprobformat(p,cf)
 elseif (probformat="") and mpt_probsh_ori:
   image (draw p withcolor cf mpt_po_)
 else:
   formatprobbox(p,cl,cb,cf)
 fi
enddef;

vardef formatnodeev(expr p)=
save pp;path pp;
pp:=nodeshape(p);
 save cl,cb,cf;
 color cl,cb,cf;
 cl:=nodelinecolor;
 cb:=nodebgcolor;
 cf:=nodefgcolor;
 if nodelinewidth=-1:
    nodelinewidth:= if mpt_isgraph: 1 else: 0.5 fi
 fi;
 if (not mpt_isgraph) and (nodeformat="") and mpt_nodesh_ori:
   image (draw p withcolor cf mpt_po_)
 else:
   formatnodebox(p,cl,cb,cf)
 fi
enddef;



vardef formatnodeleave(expr p)=
 save cl,cb,cf;
 color cl,cb,cf;
 cl:=leavelinecolor;
 cb:=leavebgcolor;
 cf:=leavefgcolor;
 if leaveformat="none":
   image (draw p withcolor cf)
 elseif leaveformat="":
   formatnodeev(p)
 else:
   formatleavebox(p,cl,cb,cf)
 fi
enddef;




vardef formatnode(expr p)=
  if known exist.arbre[colonne_cour+1][counttmp]:
    formatnodeev(p)
  else:
    formatnodeleave(p)
  fi
enddef;


boolean mpt_nodesh_ori,mpt_probsh_ori;
mpt_nodesh_ori:=false;
mpt_probsh_ori:=false;


% Forme du chemin autour de la oproba
vardef probshape(expr p)=
 mpt_probsh_ori:=true;
 if probformat="superellipse":
   superellipsebox(p)
 elseif probformat="circle":
   circlebox(p)
 elseif probformat="square":
   squarebox(p)
 else:
   bbox p %shifted -center bbox p
 fi
enddef;

% Forme du chemin autour du noeud
vardef nodeshape(expr p)=
 mpt_nodesh_ori:=true;
 if nodeformat="superellipse":
   superellipsebox(p)
 elseif (nodeformat="circle") or (nodeformat=""):
   circlebox(p)
 elseif nodeformat="square":
   squarebox(p)
 else:
   bbox p %shifted -center bbox p
 fi
enddef;

vardef leaveshape(expr p)=
 if leaveformat="superellipse":
   superellipsebox(p)
 elseif (leaveformat="circle"):
   circlebox(p)
 elseif leaveformat="square":
   squarebox(p)
 else:
   bbox p %shifted -center bbox p
 fi
enddef;





vardef formatnodepath(expr p)=
 nodeshape(p)
enddef;

%%% Prints events

vardef theevlabel(expr s,z,inc)=
  save d; numeric d;
  d=0.5*longdir(s,inc);
  s shifted (z - center s + (d+labeloffset)*dir(inc))
enddef;

def evlabel = draw theevlabel enddef;



vardef thelabelbranchehaut(expr Ori,Fin,Fig)=
   interim labeloffset:=proboffset;
   save M,pp,tt;
   pair M; path pp;numeric tt;
   pp=dessinbranche(Ori,Fin);
   tt = arctime posprob*(arclength  pp) of pp;
   M= point tt of pp;
   M:=M + 0.5(llcorner Fig - lrcorner Fig);
   if (angle(direction tt of pp) <= 90) and (angle(direction tt of pp) > -90):
      M:=M+proboffset*unitvector((direction tt of pp) rotated 90)
   else:
      M:=M-proboffset*unitvector((direction tt of pp) rotated 90)
   fi;
   Fig shifted M
enddef;


vardef thelabelbranchehautrot(expr Ori,Fin,Fig)=
   interim labeloffset:=proboffset;
   save M,pp,tt;
   pair M; path pp;numeric tt;
   pp=dessinbranche(Ori,Fin);
   tt = arctime posprob*(arclength  pp) of pp;
   M= point tt of pp;
   if  xpart Ori <= xpart Fin:
      Fig shifted ( - 0.5*lrcorner Fig- 0.5* llcorner Fig+(0,proboffset)) rotated angle(direction tt of pp) shifted M
   else:
      Fig shifted ( - 0.5*lrcorner Fig- 0.5* llcorner Fig+(0,proboffset)) rotated (-180+angle(direction tt of pp)) shifted M
   fi
enddef;



vardef thelabelbranchesuper(expr Ori,Fin,Fig)=
   save figtmp,M,pp;
   pair M; path pp;
   picture figtmp;
   pp=dessinbranche(Ori,Fin);
   M= point (arctime posprob*(arclength  pp) of pp) of pp;
   figtmp:=Fig shifted (M-0.5*(lrcorner Fig + ulcorner Fig));
   image(fill bbox figtmp withcolor background;
         draw figtmp)
enddef;


vardef thelabelbranchesuperrot(expr Ori,Fin,Fig)=
   save figtmp;
   picture figtmp;
   save M,pp,tt;
   pair M; path pp;numeric tt;
   pp=dessinbranche(Ori,Fin);
   tt = arctime posprob*(arclength  pp) of pp;
   M= point tt of pp;
   figtmp:=Fig shifted (-0.5*(lrcorner Fig + ulcorner Fig)) rotated angle(direction tt of pp) shifted M;
   image(fill bbox figtmp withcolor background;
         draw figtmp)
enddef;



vardef thelabelbranche(expr Ori,Fin,Fig)=
  if typeprob=1:
     thelabelbranchehaut(Ori,Fin,Fig)
  elseif typeprob=2:
     thelabelbranchesuper(Ori,Fin,Fig)
  elseif typeprob=3:
     thelabelbranchehautrot(Ori,Fin,Fig)
  elseif typeprob=4:
     thelabelbranchesuperrot(Ori,Fin,Fig)
  fi
enddef;

vardef brokenline(expr Ori,Fin)=
   save B,M;
   pair B,M;
   B=(xpart (Fin rotated -dirlabel), ypart (Ori rotated -dirlabel));
   M = (brokenlineratio*abs(B-(Ori rotated -dirlabel)),0) rotated dirlabel;
   Ori -- (Ori + M) -- (Fin - M) -- Fin
enddef;

numeric tenscurve;
tenscurve:=0;

vardef curveline(expr Ori,Fin)=
  save tmp_angle;
  numeric tmp_angle;
    tmp_angle:=dirlabel+tenscurve*angle((Fin-Ori) rotated -dirlabel);
  Ori{dir tmp_angle}..{dir tmp_angle}Fin
enddef;



%%%%%%%% Dessin de la branche

vardef dessinbranche(expr Ori,Fin)=
  edgeshape(Ori,Fin)
enddef;


vardef dessinarete(expr Ori,Fin)=
   save an,dec;
   numeric an;pair decdeb,decfin;
   an:=angle(Fin-Ori);
   if edgeshift<>0:
      startedgeshift:=edgeshift;endedgeshift:=edgeshift
   fi;
   decdeb:=(0,startedgeshift) rotated an;
   decfin:=(0,endedgeshift) rotated an;
   edgeshape(Ori+decdeb,Fin+decfin)
enddef;



vardef edgeshape(expr Ori,Fin)=
   save an;
   numeric an;
  if mpt_isgraph:
    an:=angle(Fin-Ori);
    (Ori){dir (an+edgeangleaa[0])}..{dir (an-edgeangleaa[1])}(Fin)
  elseif branchtype="segment":
    Ori--Fin
  elseif branchtype="curve":
    curveline(Ori,Fin)
  elseif branchtype="broken":
    brokenline(Ori,Fin)
  else:
    Ori--Fin
  fi
enddef;


%%%%%%%% Dessin complet de la branche (avec proba, évènement)

vardef branche_abs(expr Ori,Fin,Eve,Pro)=
   save figtmp,thelab,figlab,tmpshft;
   picture figtmp,thelab,figlab;
   numeric tmpshft;
   thelab:=labelarbres(Pro) scaled scaleprob;
   if ypart (Ori rotated -dirtree) > ypart (Fin rotated -dirtree):  %le décalage se fait en fonction de la direction de la branche
      tmpshft:=endedgeshift
   else:
      tmpshft:=-endedgeshift
   fi;
   figlab:=thelabelbranche(Ori,(Fin+ (0,tmpshft)),thelab);
   figtmp=image(%
      if branchtype<>"none":
         if edgearrow:
            drawarrow
         else:
            draw
         fi
         dessinbranche(Ori,(Fin + (0,tmpshft))) withpen pencircle scaled linewidth withcolor linecolor;
      fi
      draw figlab;
      evlabel(Eve,Fin,dirlabel)
      );
   figtmp
enddef;



vardef branche_rel(expr Ori,dx,dy,Eve,Pro)=
  branche_abs(Ori,Ori+(dx,dy),Eve,Pro)
enddef;

pair mptFinBranche[][]; % Branche qui mène au noeud [][]
mptFinBranche[1][1]:=origin;

pair mptFinBrancheRel[][][]; % k-ieme branche qui part de  i,j
mptFinBrancheRel[0][1][1]:=origin;

numeric mptLargBranche[][], mptHautBranche[][]; % Branche qui mène au noeud [][]
numeric mptNbBranches[][]; % Nombre de branches de l'arbre [][]
for i=0 upto 10: for j=0 upto 100: mptNbBranches[i][j]:=1; endfor endfor
mptNbBranches[0][1]:=1;


numeric mptAbs[]; % Abscisse noeud i
mptAbs[1]:=0;

pair mptchild[][][];
mptchild[0][1][1]:=(1,1);

color mptparent[][];
mptparent[1][1]:=(0,1,1);


vardef _arbre_(expr Ori)(text t)(text r)= % (pt)(pt1,pt2...)(ev,proba,ev,proba...)
  save figtmp,n,Fintmp,labeltmp,counttmp,countlab,tmpcountligne;
  picture figtmp,labeltmp[];
  numeric counttmp,countlab,tmpcountligne;
  pair Fintmp[];
  mpt_isgraph:=false;
  if dirtree<>0: dirlabel:=dirtree fi;
  if colonne_cour=1:
       countligne[1]:=1;countligne[0]:=1;
       for i=2 upto 128: countligne[i]:=0; endfor
  fi
  tmpcountligne:=countligne[colonne_cour+1];
  counttmp:=tmpcountligne;
  for i=t:
      counttmp:=counttmp+1;
      countligne[colonne_cour+1]:=countligne[colonne_cour+1]+1;
      mptFinBrancheRel[colonne_cour][ligne_cour][counttmp-tmpcountligne]:=i rotatedaround(Ori,dirtree);
      mptFinBranche[colonne_cour+1][counttmp]:=i rotatedaround(Ori,dirtree);
      mptLargBranche[colonne_cour+1][counttmp]:= xpart i - xpart Ori;
      mptHautBranche[colonne_cour+1][counttmp]:= ypart i - ypart Ori;
      mptchild[colonne_cour][ligne_cour][counttmp-tmpcountligne]:=(colonne_cour+1,counttmp);
      mptparent[colonne_cour+1][counttmp]:=(colonne_cour,ligne_cour,counttmp-tmpcountligne);
  endfor;
  mptAbs[colonne_cour+1]:= xpart mptFinBranche[colonne_cour+1][counttmp];
  mptNbBranches[colonne_cour][ligne_cour]:=counttmp-tmpcountligne;
  figtmp=image(%
    counttmp:=tmpcountligne;countlab:=0;
    for i=r:
      countlab:=countlab+1;
      labeltmp[countlab]:=labelarbres(i);
      if countlab=2:
          counttmp:=counttmp+1;
          labeltmp[1]:=formatnode(labeltmp[1]);
          labeltmp[2]:=formatprob(labeltmp[2]);
          draw branche_abs(Ori,mptFinBranche[colonne_cour+1][counttmp],labeltmp[1] scaled scaleev,labeltmp[2]);
          Orig_arbre[colonne_cour+1][counttmp]:=mptFinBranche[colonne_cour+1][counttmp] shifted
              if shiftev<0: ((longdir(labeltmp[1] scaled scaleev,dirlabel) + 2labeloffset)*(1,0))
              else: (shiftev,0)
              fi
                  rotatedaround(mptFinBranche[colonne_cour+1][counttmp],dirlabel);
         if mpt_passe=1:
           mpt_dec_ori[colonne_cour+1][counttmp]:= (Orig_arbre[colonne_cour+1][counttmp] rotatedaround(Ori,-dirtree))- mptFinBranche[colonne_cour+1][counttmp]
         fi;
         countlab:=0
      fi;
    endfor
      );
   figtmp
enddef;





vardef arbre_decv(expr Ori,horiz,Ymax)(text t)(text r)=
          % (pt,dec horiz)(dy1,dy2...)(ev,proba,ev,proba...)
  save Fin,compt,Ytmp;
  numeric compt,Ytmp;
  pair Fin[];
  if abscoord:
    Fin[0]:=(horiz,Ymax)
  else:
    Fin[0]:=Ori+(horiz,Ymax)
  fi;
  compt:=0;Ytmp:=Ymax;
  for i=t:
    compt:=compt+1;Ytmp:=i;
    if abscoord:
      Fin[compt]:=(horiz,Ytmp)
    else:
      Fin[compt]:=Ori+(horiz,Ytmp)
    fi;
  endfor
  def ptarr=
    Fin[0]
    for i=1 upto compt:
     ,Fin[i]
    endfor
  enddef;
  _arbre_(Ori)(ptarr)(r)
enddef;


vardef arbre_fixe(expr Ori,horiz,Ymax,decy)(text r)=
  save Fin,compt,compt_par;
  numeric compt,compt_par;
  pair Fin[];
  if abscoord:
    Fin[0]:=(horiz,Ymax + ypart Ori)
  else:
    Fin[0]:=Ori+(horiz,Ymax)
  fi;
  compt:=0;compt_par:=0;
  for i=r:
    compt:=compt+1;compt_par:=compt_par+1;
    if compt_par=2:
       compt_par:=0;
       if abscoord:
         Fin[compt/2]:=(0, ypart Ori) + (horiz,Ymax-decy*compt/2)
       else:
         Fin[compt/2]:=Ori+(horiz,Ymax-decy*compt/2)
       fi;
    fi
  endfor
  def ptarr=
    Fin[0]
    for i=1 upto compt/2-1:
     ,Fin[i]
    endfor
  enddef;
  _arbre_(Ori)(ptarr)(r)
enddef;


vardef colonne.reg.arbre[]=
  @
enddef;

vardef colonne.comp.arbre[]=
  @
enddef;

vardef colonne.dec.arbre[]=
  @
enddef;

vardef colonne.calc.arbre[]=
  @
enddef;

vardef colonne.arbre[]=
  @
enddef;

vardef reg.arbre[][](text t)(text r)=
 save compt,decal;
 numeric compt,decal[];
 colonne_cour:=colonne.#@;
 ligne_cour:=@;
 compt:=0;
   for i=t:
    if numeric i:
       compt:=compt+1;
       decal[compt]:=i
    fi;
   endfor
 compt:=0;
   for i=r:
    compt:=compt+1;
   endfor
 arbre_fixe(Orig_arbre[colonne_cour][ligne_cour],decal[1],(compt/2-1)*decal[2]/2,decal[2])(r)
enddef;



vardef comp.arbre[][](text t)(text r)=
  colonne_cour:=colonne.#@;
  ligne_cour:=@;
  save tmp_compt;
  numeric tmp_compt;
  tmp_compt:=0;
  if abscoord:
    def tmp_text=t enddef
  else:
  def tmp_text= % Si abscoord tmp_text=t
    for i=t:
      hide(tmp_compt:=tmp_compt+1;)
      if (tmp_compt>1) and (not string i):
         ,Orig_arbre[colonne_cour][ligne_cour]+i
      elseif not string i:
         Orig_arbre[colonne_cour][ligne_cour]+i
      fi
    endfor
  enddef
  fi;
  _arbre_(Orig_arbre[colonne_cour][ligne_cour])(tmp_text)(r)
enddef;


%%%%%%%%%%%%%%%%%%%%%%%%% Modif 10/16 - Décalage par rapport à l'origine
vardef dec.arbre[][](text t)(text r)=
  save decal;
  numeric decal[];
  colonne_cour:=colonne.#@;
  ligne_cour:=@;
  save tmp_compt;
  numeric tmp_compt;
  tmp_compt:=0;
  for i=t:
    if numeric i:
       tmp_compt:=tmp_compt+1;
       decal[tmp_compt]:=i
    fi;
  endfor
  def tmp_text=
    decal[3]
    for i=4 upto tmp_compt:
      ,decal[i]
    endfor
  enddef;
  arbre_decv(Orig_arbre[colonne_cour][ligne_cour],decal[1],decal[2])(tmp_text)(r)
enddef;


vardef calc.arbre[][](text r)=
  colonne_cour:=colonne.#@;
  ligne_cour:=@;
  save mpt_O,mpt_dec_calc;
  pair mpt_O;
  numeric mpt_dec_calc[];
   for i=1 upto mptNbBranches[colonne_cour][ligne_cour]:
     mpt_dec_calc[i]:=mpt_decbranche[colonne_cour][ligne_cour][i];
   endfor
   mpt_dec_calc[0]:=widthbranch*(scalebranch**colonne_cour);
  def tmp_text_calc=
    mpt_dec_calc[0]
   for i=1 upto mptNbBranches[colonne_cour][ligne_cour]:
     ,mpt_dec_calc[i]
   endfor
  enddef;
   mpt_O:= mpt_tmp_ori[colonne_cour][ligne_cour];
   dec.arbre[colonne_cour][ligne_cour](tmp_text_calc)(r)
%   _arbre_(mpt_O)(tmp_text_calc)(r)
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Macro  principale - Modification Mars 2022 (variables locales)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Conversion des ' en " pour les variables locales.

vardef singletodoublequote(expr chdb)=
  save chsi;
  string chsi;
  chsi:="";
  for i=1 upto length(chdb):
    if ASCII(substring(i-1,i) of chdb) = 39:
      chsi := chsi & char(34)
    else:
      chsi := chsi & substring(i-1,i) of chdb
    fi;
  endfor
  chsi
enddef;

% Initialisation variables locales

def init_loc_var=
  numeric tmp_posprob, tmp_shiftev, tmp_typeprob, tmp_scaleev, tmp_scaleprob, tmp_proboffset, tmp_endedgeshift;
  boolean tmp_edgearrow;
  string tmp_branchtype;
  tmp_posprob := posprob;
  tmp_typeprob := typeprob;
  tmp_shiftev:= shiftev;
  tmp_scaleev := scaleev;
  tmp_scaleprob := scaleprob;
  tmp_proboffset := proboffset;
  tmp_endedgeshift := endedgeshift;
  tmp_edgearrow := edgearrow;
  tmp_branchtype := branchtype;
  save posprob, shiftev, typeprob, scaleev, scaleprob, proboffset, endedgeshift, edgearrow, branchtype;
  numeric posprob, shiftev, typeprob, scaleev, scaleprob, proboffset, endedgeshift;
  boolean edgearrow;
  string branchtype;
  posprob:=tmp_posprob;
  typeprob:=tmp_typeprob;
  shiftev:=tmp_shiftev;
  scaleev := tmp_scaleev;
  scaleprob := tmp_scaleprob;
  proboffset := tmp_proboffset;
  endedgeshift := tmp_endedgeshift;
  edgearrow := tmp_edgearrow;
  branchtype := tmp_branchtype;
enddef;


pair current_tree;

boolean arbrebool; % est-on à l'intérieur d'un arbre ?
arbrebool:=false;

vardef arbre[][](text t)(text r)=
  arbrebool:=true;
  naturalwidth:=false;
  init_loc_var;
  if posprob=-1: posprob:=0.6 fi;
  save tmp_type,tmp_compt,tmp_chdb;
  numeric tmp_type,tmp_compt,tmp_comptii;
  string tmp_chdb;
  tmp_compt:=0;tmp_comptii:=0;
  for i=t:
    tmp_comptii:=tmp_comptii+1;
    if numeric i:
       tmp_compt:=tmp_compt+1
    elseif string i:
       tmp_chdb:=singletodoublequote(i);
       scantokens tmp_chdb;
    fi;
  endfor
  exist.#@.@:=true;
  current_tree:=(colonne.#@,@);
  if tmp_comptii=0:
   if mpt_passe=1:
    reg.#@.@(widthbranch*(scalebranch**colonne_cour),gapnode*(2**(3-colonne_cour)))(r)
   else:
    calc.#@.@(r)
   fi
  else:
     Orig_arbre[1][1]:=origin;
     if tmp_compt=0: % Arbre défini par des points (dec_horiz,dec_vert)
        comp.#@.@(t)(r)
     elseif tmp_compt=2: % Arbre régulier, on donne largeur et esp. vert.
        reg.#@.@(t)(r)
     else:
        dec.#@.@(t)(r) % Arbre semi-régulier, largeur fixe, decalages vert.
     fi
  fi
enddef;


def tree = arbre enddef;

%%%%% stree (simple tree) Arbre sans les probas

vardef colonne.stree[]=
  @
enddef;

vardef stree[][](text t)(text r)=
 save cc,ll;
 numeric cc,ll;
 cc:=colonne.#@;
 ll:=@;
 def tmptxt=
    for i=r:
      i,"",
    endfor
 enddef;
 tree[cc][ll](t)(tmptxt)
enddef;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% "Calculated" trees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vardef ymean(expr i,j)=
  (0
   for k=1 upto mptNbBranches[i][j]:
    + ypart (mptFinBrancheRel[i][j][k] rotated -dirtree)
  endfor)/mptNbBranches[i][j]
enddef;


vardef barycentre(expr A,a,B,b)=
  (a*A+b*B)/(a+b)
enddef;

numeric mpt_fact_gap;
mpt_fact_gap:=1.5;

vardef exlignes(expr col,lig)=
% renvoie (x,y) tel que x<lig<y, x et y ont des child, 0 sinon
  save tmpcl,tmpx,tmpy,ytrue;
  pair tmpcl;
  numeric tmpx,tmpy;
  boolean ytrue;
  tmpx:=0;tmpy:=0;ytrue:=false;
  for i=1 upto countligne[col+1]:
    if known exist.arbre[col+1][i]:
       if i<lig:
           tmpx:=i
       else:
           tmpy:=i;ytrue:=true;
       fi
    fi;
  exitif ytrue;
  endfor
  (tmpx,tmpy)
enddef;


vardef mptCalcIsol(expr a,b,c,l)=
  save tmpmin,tmpmax;
  numeric tmpmin,tmpmax;
  tmpmin:=xpart exlignes(a,l);
  tmpmax:=ypart exlignes(a,l);
  if tmpmin=0:
    mptFinBrancheRel[a][greenpart mptparent[a+1][tmpmax]][bluepart mptparent[a+1][tmpmax]] + (tmpmax-l)*gapnode*mpt_fact_gap*((0,1) rotated dirtree)
  elseif tmpmax=0:
    mptFinBrancheRel[a][greenpart mptparent[a+1][tmpmin]][bluepart mptparent[a+1][tmpmin]] - (l-tmpmin)*gapnode*mpt_fact_gap*((0,1) rotated dirtree)
  else:
    barycentre(mptFinBrancheRel[a][greenpart mptparent[a+1][tmpmin]][bluepart mptparent[a+1][tmpmin]],tmpmax-l,mptFinBrancheRel[a][greenpart mptparent[a+1][tmpmax]][bluepart mptparent[a+1][tmpmax]],l-tmpmin)
  fi
enddef;


numeric tmplig,tmpligii;

numeric mpt_decbranche[][][];



def AdjustNodes=
  tmplig:=0;
  for i=1 upto countligne[colonne_cour]:
    for j= 1 upto mptNbBranches[colonne_cour][i]:
      mptFinBrancheRel[colonne_cour][i][j]:=mptFinBranche[colonne_cour+1][1] -tmplig*gapnode*((0,1) rotated dirtree);
      tmplig:=tmplig+1;
    endfor
  endfor
 for cmpttmp:=1 upto colonne_cour-1:  % plusieurs  tours pour tout régler...
  for col=colonne_cour-1 downto 0:
    tmplig:=0;
    for i=1 upto countligne[col]:
     if known mptNbBranches[col][i]:
      for j= 1 upto mptNbBranches[col][i]:
         if known exist.arbre[col][i]:
             tmplig:=tmplig+1;
           if known exist.arbre[col+1][tmplig]:
              mptFinBrancheRel[col][i][j]:=
                       (xpart (mptFinBrancheRel[col][i][j] rotated -dirtree),ymean(col+1,tmplig)) rotated dirtree;
           else:
              mptFinBrancheRel[col][i][j]:=mptCalcIsol(col,i,j,tmplig);
           fi;
           Orig_arbre[col+1][tmplig]:=(xpart (Orig_arbre[col+1][tmplig] rotated -dirtree),ypart (mptFinBrancheRel[col][i][j] rotated -dirtree)) rotated dirtree;
         fi;
      endfor
     fi
    endfor
  endfor
%  adjustii;
 endfor
  for col=1 upto colonne_cour:
    for i=1 upto countligne[col]:
      if known exist.arbre[col][i]:
        for j=1 upto mptNbBranches[col][i]:
           mpt_decbranche[col][i][j]:=ypart (mptFinBrancheRel[col][i][j] rotated -dirtree) - ypart (Orig_arbre[col][i] rotated -dirtree);
        endfor
      fi
    endfor
  endfor
  Orig_arbre[1][1]:=origin;
enddef;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% regular trees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numeric scalebranch;
scalebranch := 0.8;

numeric widthbranch,gapnode;
widthbranch:=3.5cm;
gapnode:=0.7cm;

vardef regulartree(expr n)(text t)(text r)=
  save tmp,ll,shiftev,tmpstr,count,cc,tmp_compt,h;
  numeric ll,count,cc,tmp_compt,h;
  string tmpstr[][];
  count:=0;cc:=1;
  for i = r:
    count := count + cc;
    cc := 1 - cc;
    tmpstr[count][cc] := i;
  endfor
  shiftev:=-1;
  for j=1 upto count:
    if shiftev < longdir(labelarbres(tmpstr[j][0]),dirtree):
      shiftev := longdir(labelarbres(tmpstr[j][0]),dirtree)
    fi;
  endfor
  shiftev := shiftev + 2*labeloffset;
  tmp_compt:=1;
  for i=t:
    if (not string i) and (tmp_compt=1):
       ll := i; tmp_compt:=2
    elseif (not string i) and (tmp_compt=2):
       h := i
    else:
      scantokens i
    fi;
  endfor
  picture tmp;
  tmp := image(%
  for i := 1 upto n:
    for j := 1 upto (count**(i-1)):
      draw tree[i][j](ll,h*(count**(n-i)))(r);
    endfor
    ll := ll * scalebranch;
  endfor
  );
  tmp
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Binomial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

string bernoullisuccessevent, bernoullifailureevent, bernoullisuccessprob, bernoullifailureprob;
bernoullisuccessevent := "$S$";
bernoullifailureevent := "$\overline{S}$";
bernoullisuccessprob := "$p$";
bernoullifailureprob := "$q$";
numeric bernoulliscalebranch;
bernoulliscalebranch := 0.8;

vardef bernoulliprocess(expr n)(text t)(text r)=
  save tmp_ct,tmp_ch;
  numeric tmp_ct;
  string tmp_ch[];
  tmp_ch[1]:=bernoullisuccessevent;
  tmp_ch[2]:=bernoullisuccessprob;
  tmp_ch[3]:=bernoullifailureevent;
  tmp_ch[4]:=bernoullifailureprob;
  tmp_ct:=0;
  for i=r:
    tmp_ct:=tmp_ct+1;
    if string i:
      tmp_ch[tmp_ct]:=i
    fi;
  endfor
  regulartree(n)(t)(tmp_ch[1],tmp_ch[2],tmp_ch[3],tmp_ch[4])
enddef;



vardef bernoulliprocessL(expr n)(text t)(text r)=
  save L,H,l,hh,lpic,hpic;
  numeric L,H,l,hh,lpic,hpic;
  save tmp_ct,tmp_chi,tmp_b;
  numeric tmp_ct;
  string tmp_chi[];
  boolean tmp_b;
  tmp_chi[1]:=bernoullisuccessevent;
  tmp_chi[2]:=bernoullisuccessprob;
  tmp_chi[3]:=bernoullifailureevent;
  tmp_chi[4]:=bernoullifailureprob;
  tmp_b:=true;
  for i=t:
    if numeric i:
      if tmp_b: L:=i; tmp_b := false else: H:=i fi
    fi;
  endfor
  tmp_ct:=0;
  for i=r:
    tmp_ct:=tmp_ct+1;
    if string i:
      tmp_chi[tmp_ct]:=i
    fi;
  endfor
  hpic:=0.5*longdir(labelarbres(tmp_chi[1]),90+dirtree) + 0.5*longdir(labelarbres(tmp_chi[3]),90+dirtree);
  hh := (H-hpic)/((2**n)-1);
  lpic:=longdir(mptreestartlabel,dirtree) + n*max(longdir(labelarbres(tmp_chi[1]),dirtree) , longdir(labelarbres(tmp_chi[3]),dirtree)) + (2*n-1)*labeloffset;
  if bernoulliscalebranch<>1:
    l:= (L-lpic)*(1-bernoulliscalebranch)/(1-bernoulliscalebranch**n)
  else:
    l:= (L-lpic)/n
  fi;
  def tmp_text=
    l,hh
      for i=t:
        if string i: ,i fi
      endfor
  enddef;
  regulartree(n)(tmp_text)(tmp_chi[1],tmp_chi[2],tmp_chi[3],tmp_chi[4])
enddef;


vardef binomialtree(expr n)(expr l,h)=
  save tmp,ll,posprob,shiftev,bernoulliscalebranch;
  numeric ll;
  posprob:=0.5;bernoulliscalebranch:=1;
  shiftev:=longdir(labelarbres("3"),dirtree)+2*labeloffset;
  ll := l;
  picture tmp;
  tmp := image(%
  draw tree[1][1](ll,h)("$1$", bernoullisuccessprob,  "$0$", bernoullifailureprob);
  if n>1:
    for i := 2 upto n:
      for j := 1 upto (i-1):
        draw tree[i][2*j-1](ll,h)(decimal(i-j+1), bernoullisuccessprob,  " ", bernoullifailureprob);
      endfor
      draw tree[i][2*i-2](ll,h)("$1$", bernoullisuccessprob,  "$0$", bernoullifailureprob);
    endfor
    ll := ll * bernoulliscalebranch;
  fi
  );
  tmp
enddef;

vardef binomialtreeL(expr n)(expr L,H)=
  save l,h,lpic,hpic;
  numeric l,h,lpic,hpic;
  hpic:=longdir(labelarbres("0"),90+dirtree);
  h := (H-hpic)/n;
  lpic:=longdir(mptreestartlabel,dirtree) + n*max(longdir(labelarbres("0"),dirtree) , longdir(labelarbres("0"),dirtree)) + (2*n-1)*labeloffset;
  l:= (L-lpic)/n;
  binomialtree(n)(l,h)
enddef;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         GRAPHES
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%
%% Paramètres
%%%%%


numeric nodewidth,edgelinewidth,edgearrowpos,edgeshift,startegdeshift;
nodewidth:=0.6cm;
edgelinewidth:=1;
edgearrowpos:=1;
edgeshift:=0;
startedgeshift:=0;

boolean naturalwidth;
naturalwidth:=false;

numeric nodelabeloffset, nodeedgeoffset; % pour la position des étiquettes
nodelabeloffset:=2;
nodeedgeoffset:=-101;



numeric edgeangle,loopangle,loopsize,loopstartangle;
edgeangle:=0;
loopangle:=90;
loopsize:=-1;
loopstartangle:=30;


%%% Format du nom des sommets
string defnodename;
defnodename:="A";
string nodename;
nodename:="array";
numeric startnodename;
startnodename=1;


boolean printnodename;
printnodename:=true;



numeric mpt_numtotal; % nombre total de sommets
mpt_numtotal:=0;

pair mpt_nodecoord[]; % coordonnées du noeud i
string mpt_nodename[]; % nom du noeud i
path mpt_nodepath[]; % chemin autour du noeud i
string mpt_nodelpart[],mpt_nodenumpart[]; % parties alphanumérique et numérique du nom du noeud

color gridcolor;
gridcolor:=0.7white;

numeric gridlinewidth;
gridlinewidth:=0.5;



%%%% Variables privées
boolean mpt_dir;  %arête orientée ou non
mpt_dir:=false;

boolean mpt_isgraph;
mpt_isgraph:=false;

picture mpt_tmp_fig_cbg;
path mpt_tmp_branche;
string mpt_tmp_prob;

path mpt_tmp_path;

boolean mpt_isprob,mpt_isprobover;
mpt_isprob:=false;
mpt_isprobover:=false;


boolean mpt_bgc;
mpt_bgc:=false;





%%%%%%%%%%%%%%%%%%%%%%%%%%%% Noeuds

vardef mptlpart(expr tt)=
  save tmp;
  string tmp;
  tmp:="";
  for i =1 upto length tt:
    if not isdigit substring(i-1,i) of tt:
      tmp:=tmp & substring(i-1,i) of tt
    fi;
  endfor
  tmp
enddef;

vardef mptnumpart(expr tt)=
  save tmp;
  string tmp;
  tmp:="";
  for i =1 upto length tt:
    if isdigit substring(i-1,i) of tt:
      tmp:=tmp & substring(i-1,i) of tt
    fi;
  endfor
  tmp
enddef;

% définit le noeud et certains attributs
def defnode(suffix ptA)(expr A)=
  save tmpchcomp,tmpchlet,tmpchnum;
  string tmpchcomp,tmpchlet,tmpchnum;
   mpt_isgraph:=true;arbrebool:=false;
  tmpchcomp:= str ptA;
  tmpchlet:=mptlpart(str ptA);
  tmpchnum:=mptnumpart(str ptA);
  if tmpchnum="":
     pair ptA;
     ptA:=A;
  else:
     scantokens("if not known " & tmpchlet & ": pair " & tmpchlet & "[] fi;" & tmpchlet & tmpchnum & ":=("& decimal(xpart A)&","& decimal(ypart A) & ");")
  fi;
  mpt_numtotal:=mpt_numtotal+1;
  mpt_nodecoord[mpt_numtotal]:=A;
  mpt_nodename[mpt_numtotal]:= tmpchcomp;
  mpt_nodelpart[mpt_numtotal]:= tmpchlet;
  mpt_nodenumpart[mpt_numtotal]:= tmpchnum;
  mpt_nodepath[mpt_numtotal]:= formatnodepath(labelarbres(str ptA) scaled scaleev) shifted A;
enddef;


def defnodestr(expr ptA)(expr A)=   % Quand le premier paramètre est une chaine simple
  save tmpchcomp,tmpchlet,tmpchnum;
  string tmpchcomp,tmpchlet,tmpchnum;
  mpt_isgraph:=true;arbrebool:=false;
  tmpchcomp:= ptA;
  tmpchlet:=ptA;
  tmpchnum:="";
  scantokens("pair " & tmpchcomp & ";" & tmpchcomp & ":=("& decimal(xpart A)&","& decimal(ypart A) & ");")
  mpt_numtotal:=mpt_numtotal+1;
  mpt_nodecoord[mpt_numtotal]:=A;
  mpt_nodename[mpt_numtotal]:= tmpchcomp;
  mpt_nodelpart[mpt_numtotal]:= tmpchlet;
  mpt_nodenumpart[mpt_numtotal]:= tmpchnum;
  mpt_nodepath[mpt_numtotal]:= formatnodepath(labelarbres(tmpchcomp) scaled scaleev) shifted A;
enddef;


def defnodes(text t)=   % Pour définir plusieurs noeuds à la fois
 mpt_isgraph:=true;arbrebool:=false;
 if (nodename="array") or (nodename="arabic"):
  for i=t:
    mpt_numtotal:=mpt_numtotal+1;
    if mpt_numtotal=1:
       scantokens("pair " & defnodename & "[];")
    fi;
    scantokens(defnodename & decimal(mpt_numtotal) & ":=("& decimal(xpart i)&","& decimal(ypart i) & ");");
    mpt_nodecoord[mpt_numtotal]:=i;
    mpt_nodename[mpt_numtotal]:= defnodename & decimal(mpt_numtotal);
    mpt_nodelpart[mpt_numtotal]:= defnodename;
    mpt_nodenumpart[mpt_numtotal]:= decimal(mpt_numtotal);
    mpt_nodepath[mpt_numtotal]:= formatnodepath(labelarbres(mpt_nodename[mpt_numtotal]) scaled scaleev) shifted mpt_nodecoord[mpt_numtotal];
  endfor
 elseif (nodename="Alph") or (nodename="alph"):
  save tmp_str,tmp_nn;
  string tmp_str;
  numeric tmp_nn;
  tmp_nn:= if nodename="Alph": 64 else: 96 fi;
  for i=t:
    tmp_str:=char(mpt_numtotal+startnodename+tmp_nn);
    defnodestr(tmp_str)(i);
  endfor
 fi
enddef;


% Renvoie le rang d'un noeud par suffixe
vardef searchnode(suffix ptA)=
  save val,st;
  numeric val;
  string st;
  st= str ptA;
  for i=1 upto mpt_numtotal:
     if mpt_nodename[i]=st:
       val=i
     fi;
     exitif known val;
  endfor
  val
enddef;

% Renvoie le rang d'un noeud suivi d'une chaine
vardef searchnodech(suffix s)(expr c)=
  searchnode(s)
enddef;

% Renvoie le path d'un noeud
vardef nodepath(suffix ptA)=
  save nn;
  numeric nn;
  nn:=searchnode(ptA);
  mpt_nodepath[nn]
enddef;



% Dessin du noeud
vardef thenode@#(text t)=
  save tmp_fig,tmpstr;
  picture tmp_fig;
  string tmp_str;
  if long_texte(t)=1:
     tmp_fig:=thenodenum@#(searchnode(t))
  else:
     for i=t:
       if string i: tmp_fig:=thenodenum@#(searchnodech(t),i) fi;
     endfor
  fi;
  tmp_fig
enddef;

vardef thenodenum@#(text t)=
   save tmp_pt,ch,dess,latch,num;
   numeric num;
   pair tmp_pt;
   string ch,latch;
   picture dess;
   mpt_isgraph:=true;arbrebool:=false;
%   interim labeloffset:=nodelabeloffset+nodewidth/2;
   interim labeloffset:=nodelabeloffset;
   if ((ASCII str @# < 58) or (ASCII str @# = 91)) and (str @#<>""):
       def mylabel = labelang enddef
   else:
       def mylabel = label enddef
   fi;
   if long_texte(t)=1:
        num:=t;
        dess:=image(
          ch:="$" & _chainec(mpt_nodename[num]) & "$";
          if nodename="arabic": ch := "$" & decimal(num) & "$" fi;
          mpt_nodepath[num]:=formatnodepath(labelarbres(ch) scaled scaleev) shifted mpt_nodecoord[num];
          draw formatnodeev(labelarbres(ch) scaled scaleev) shifted mpt_nodecoord[num];
          if printnodename:
            mylabel@#(textext(ch) scaled scaleev, ptofpath@#(mpt_nodepath[num],mpt_nodecoord[num])) withcolor nodefgcolor mpt_po_
          fi)
   else:
      dess:=image(%
      for PP=t:
        if numeric PP:
           num:=PP;
           tmp_pt:=mpt_nodecoord[num];
        else:
          mpt_nodepath[num]:=formatnodepath(labelarbres(PP) scaled scaleev) shifted mpt_nodecoord[num];
          draw formatnodeev(labelarbres(PP) scaled scaleev) shifted tmp_pt;
          if printnodename:
           if string PP:
              mylabel@#(textext(PP),ptofpath@#(mpt_nodepath[num],mpt_nodecoord[num])) withcolor nodefgcolor mpt_po_;
           else:
              mylabel@#(PP scaled defaultscale,ptofpath@#(mpt_nodepath[num],mpt_nodecoord[num])) withcolor nodefgcolor mpt_po_;
           fi
          fi
        fi;
      endfor)
   fi;
   dess
enddef;

vardef node@#(text s)=
  save mpt_boolnum;
  boolean mpt_boolnum;
  mpt_boolnum:=false;
  for i=s:
    if numeric i:
       mpt_boolnum:=true
    fi;
  endfor
  if mpt_boolnum:
    thenodenum@#(s)
  else:
    thenode@#(s)
  fi
enddef;


vardef firstnode@#(suffix s)(text t) text r=
 drawnode@#(s) r;
 def tmp_text= t enddef;
enddef;


vardef nodes@#(text t) text s=  % Dessin des noeuds à partir de leur rang
  save tmp_pt;
  pair tmp_pt;
  save mpt_boolnum;
  boolean mpt_boolnum;
  mpt_boolnum:=false;
  for i=t:
    if numeric i:
       mpt_boolnum:=true
    fi;
  endfor
  image(
  if long_texte(t)=0:
     for i=1 upto mpt_numtotal:
       drawnode@#(i) s;
     endfor
  elseif mpt_boolnum:
    for i=t:
      drawnode@#(i) s;
    endfor
  else:
    save n;numeric n;
    n=long_texte(t);
    def tmp_text= t enddef;
    for i=1 upto n-1:
      firstnode@#(tmp_text) s;
    endfor
    drawnode@#(tmp_text) s;
  fi
  )
enddef;



vardef drawnode@#(text p) text t =
   mpt_drop(t);
   mpt_tmp_fig_cbg:=image(draw (0,0) mpt_po_);
   draw node@#(p);
   mpt_bgc:=false;
   mpt_drop();
enddef;

def drawnodes = draw nodes enddef;



%%%%%%%%%%%%%
%% Arêtes
%%%%%%%%%%%%%
% chemin entre les noeuds de rangs n et m, angles en param
vardef theedgeAB(expr m,n)=
  save an,ta,pb,pathab,A,B,C,tmpoffset;
  numeric an,tmpoffset;
  numeric ta,tb;
  pair A,B,C;
  if nodeedgeoffset=-101: tmpoffset:=0
  else:                   tmpoffset:=nodeedgeoffset
  fi;
  tmpoffset:=tmpoffset+nodelinewidth/2+edgelinewidth/2;
  A:=mpt_nodecoord[m];
  B:=mpt_nodecoord[n];
  path pathab;
  if m<>n:
     pathab:=dessinarete(A,B);
    ta=xpart(pathab intersectiontimes mpt_nodepath[m]);
    tb=xpart(pathab intersectiontimes mpt_nodepath[n]);
    pathab:= subpath(ta,tb) of pathab;
    ta:= arctime tmpoffset of pathab;
    tb:= arctime ((arclength pathab)-tmpoffset) of pathab;
    pathab:= subpath (ta, tb) of pathab;
  else:
    pathab:=theloopA(n)
  fi;
  pathab
enddef;



% boucle sur le noeud n
vardef theloopA(expr n)=
  save pathaa,aapath,A,B,C,tmpoffset,ta,tb;
  pair A,B,C;
  numeric tmpoffset,ta,tb;
  path pathaa,aapath;
  if nodeedgeoffset=-101: tmpoffset:=0
  else:                   tmpoffset:=nodeedgeoffset
  fi;
  tmpoffset:=tmpoffset+nodelinewidth/2+edgelinewidth/2;
  C:=mpt_nodecoord[n];
  pathaa:=C{dir (-tmp_loopangle[1]+tmp_loopangle[0])}..
          (C+((0.5loopsize,0.77loopsize) rotated (tmp_loopangle[0]-90))){dir (tmp_loopangle[0])}..
          (C+((loopsize,0) rotated (tmp_loopangle[0]))){dir (tmp_loopangle[0]+90)}..
          (C+((-0.5loopsize,0.7loopsize) rotated (tmp_loopangle[0]-90))){dir (tmp_loopangle[0]+180)}..
          {dir (tmp_loopangle[1]+tmp_loopangle[0]+180)}C;
  aapath:= reverse pathaa;
  ta=xpart(subpath(0,0.5*length pathaa) of pathaa intersectiontimes mpt_nodepath[n]);
  tb=xpart(subpath(0,0.5*length aapath) of aapath intersectiontimes mpt_nodepath[n]);
  pathaa:= subpath (ta,(length pathaa) - tb) of pathaa;
  ta:= arctime tmpoffset of pathaa;
  tb:= arctime ((arclength pathaa)-tmpoffset) of pathaa;
  pathaa:= subpath (ta, tb) of pathaa;
  pathaa
enddef;

% Chemin entre les noeuds A et B, angles dans p
vardef theedge(suffix A,B)(text p)=
  save nA,nB,tmp_fig;
  numeric nA,nB;
  picture tmp_fig;
  nA:=searchnode(A);
  nB:=searchnode(B);
  theedgenum(nA,nB)(p)
enddef;

% Chemin entre les noeuds nA et nB, angles dans p
vardef theedgenum(expr nA,nB)(text p)=
  save tmp_fig;
  picture tmp_fig;
  numeric edgeangleaa[],tmp_loopangle[];
  edgeangleaa[0]=edgeangleaa[1]=edgeangle;
  tmp_loopangle[0]:=loopangle;tmp_loopangle[1]:=loopstartangle;
  mpt_isgraph:=true;arbrebool:=false;
  nn:=0;
  for i=p:
    if numeric i:
       if nn=0: edgeangleaa[0]:=i;edgeangleaa[1]:=i;tmp_loopangle[0]:=i
       else: edgeangleaa[1]:=-i;tmp_loopangle[1]:=i
       fi;
       nn:=nn+1;
    elseif string i:
       mpt_tmp_prob:=i;
    fi
  endfor
  theedgeAB(nA,nB)
enddef;

% Figure représentant le chemin avec (ou non) la flèche
vardef theedgenumfig(expr nA,nB)(text p)=
  save tmppath;
  path tmppath;
  tmppath:=theedgenum(nA,nB)(p);
  image(draw tmppath withpen pencircle scaled edgelinewidth withcolor linecolor mpt_po_;
       if mpt_dir:
         filldraw arrowhead subpath(0, edgearrowpos*length mpt_tmp_path) of mpt_tmp_path mpt_po_
       fi;
       )
enddef;

vardef edges@#(text s)(text p) text t=  % dessine plusieurs arêtes simultanément
  save a,b,isp;
  numeric a,b;
  boolean isp;
  isp:=false;
  image(
    for i=s:
    if pair i:
      if isp: drawedge@#(a,b)(p) t; fi;
      a:= xpart i;b:= ypart i;
      isp:=true;
    else:
       drawedge@#(a,b)(i,p) t;
       isp:=false
    fi;
  endfor
  if isp: drawedge(a,b)(p) t; fi
  )
enddef;






vardef theedgepond@#(suffix A,B)(text p)=
  save nA,nB;
  numeric nA,nB;
  nA:=searchnode(A);
  nB:=searchnode(B);
  theedgepondnum@#(nA,nB)(p)
enddef;

vardef theedgepondnum@#(expr nA,nB)(text p)=
  save tmpfig,tmppath,tmppt,mpt_isprob,mpt_isprobover;
  boolean mpt_isprob,mpt_isprobover;
  mpt_isprob:=true;
  path tmppath;
  picture tmpfig;
  pair tmppt;
  if str @#="": mpt_isprobover:=true else: mpt_isprobover:=false fi;
  if ((ASCII str @# < 58) or (ASCII str @# = 91)) and (str @#<>""):
      def mylabel = labelang enddef
  else:
      def mylabel = labelmp enddef
  fi;
  tmppath:=theedgenum(nA,nB)(p);
  tmppt:=point (arctime posprob*(arclength  tmppath) of tmppath) of tmppath;
  image(draw theedgenum(nA,nB)(p) withpen pencircle scaled edgelinewidth withcolor linecolor mpt_po_;
        if mpt_dir:
           filldraw arrowhead subpath(0, edgearrowpos*length mpt_tmp_path) of mpt_tmp_path mpt_po_
        fi;
        mylabel@#(formatprob(labelarbres(mpt_tmp_prob) scaled scaleprob),tmppt))
enddef;

vardef edge@#(text s)(text p)=
  save mpt_boolprob,mpt_boolnum;
  boolean mpt_boolprob,mpt_boolnum;
  mpt_boolprob:=false;
  mpt_boolnum:=false;
  if posprob=-1: posprob:=0.5 fi;
  if loopsize=-1:
       loopsize:=1.5nodewidth;
       if loopsize<0.6cm: loopsize:=0.6cm fi;
  fi;
  for i=p:
    if string i:
      mpt_boolprob:=true
    fi;
  endfor
  for i=s:
    if numeric i:
       mpt_boolnum:=true
    fi;
  endfor
  if mpt_boolnum:
     mpt_tmp_path:=theedgenum(s)(p);
     if mpt_boolprob:
        theedgepondnum@#(s)(p)
     else:
        theedgenumfig(s)(p)
     fi
  else:
     mpt_tmp_path:=theedge(s)(p);
     if mpt_boolprob:
        theedgepond@#(s)(p)
     else:
        theedge(s)(p)
     fi
  fi
enddef;

%%%%%%%%%% Drawing edges

vardef drawedge@#(text s)(text p) text t=
   mpt_drop(t);
   mydraw edge@#(s)(p) withpen pencircle scaled edgelinewidth;
   mpt_drop();
enddef;

vardef drawedges@#(text s)(text p) text t=
  mydraw edges@#(s)(p) t
enddef;


vardef drawdiredge@#(text s)(text p) text t=
  save mpt_dir;
  boolean mpt_dir;
  mpt_dir:=true;
  drawedge@#(s)(p) t;
enddef;

vardef drawdiredges@#(text s)(text p) text t=
  save mpt_dir;
  boolean mpt_dir;
  mpt_dir:=true;
  drawedges@#(s)(p) t;
enddef;


vardef drawgraph@#(text s)(text p) text t=  % dessine les noeuds et les arêtes
  drawnodes@#() t;
  drawedges@#(s)(p) t;
enddef;

vardef drawdirgraph@#(text s)(text p) text t=  % dessine les noeuds et les arêtes
  drawnodes@#() t;
  drawdiredges@#(s)(p) t;
enddef;


%%%%%%%%%%%%%% Managing colors

def mpt_drop(text t) =
  def mpt_po_ = t enddef;
enddef;

mpt_drop();
def withbgcolor expr c=
  hide(mpt_bgc:=true; def mpt_opbg_ = withcolor c enddef)
enddef;

def mpt_opbg_ = enddef;

%%%%%%%%%%%%%% Grids

vardef thegridcomp(expr xmin,ymin,xmax,ymax,pas,u)=
  image(
  for i=xmin step pas until xmax:
     draw (i*u,ymin*u)--(i*u,ymax*u) withcolor gridcolor withpen pencircle scaled gridlinewidth;
     label.bot(textext("$" & decimal(i)& "$") scaled 0.6,(i*u,ymin*u)) withcolor gridcolor;
   endfor
  for i=ymin step pas until ymax:
     draw (xmin*u,i*u)--(xmax*u,i*u) withcolor gridcolor withpen pencircle scaled gridlinewidth;
     label.lft(textext("$" & decimal(i)& "$") scaled 0.6,(xmin*u,i*u)) withcolor gridcolor;
   endfor
  )
enddef;


vardef grid(text t)(text s)=
  save minmax,pasun,c;
  numeric minmax[],pasun[],c;
  c:=0;
  pasun[1]:=1cm;
  pasun[2]:=1;
  for i=s:
    c:=c+1;
    pasun[c]:=i;
  endfor
  c:=0;
  for i=t:
    c:=c+1;
    minmax[c]:=i;
  endfor
  if c=4:
    thegridcomp(minmax[1],minmax[2],minmax[3],minmax[4],pasun[2],pasun[1])
  else:
    thegridcomp(0,0,minmax[1],minmax[2],pasun[2],pasun[1])
  fi
enddef;

def drawgrid= draw grid enddef;



%%%%%%%%%%%%%%%
%% Outils
%%%%%%%%%%%%%%%%


vardef isdigit primary d =
    ("0"<=d)and(d<="9")
enddef ;

% Convertit un suffixe en chaine avec les chiffres en indice
vardef _chaine(suffix s)=
  save c,d;
  string c,d;
  d:= str s;
  if length(d)=1:
     c:=d
     elseif isdigit (substring (1,2) of d):
            c:= (substring (0,1) of d) & "_{" & substring (1,infinity) of d & "}"
            else: c:=str s
  fi;
  c
enddef;

vardef _chainec(expr d)=
  save c;
  string c;
  if length(d)=1:
     c:=d
  elseif isdigit (substring (1,2) of d):
     c:= (substring (0,1) of d) & "_{" & substring (1,infinity) of d & "}"
  elseif isdigit (substring (2,3) of d):
     c:= (substring (0,2) of d) & "_{" & substring (2,infinity) of d & "}"
  else:
     c:=d
  fi;
  c
enddef;

% longueur d'un paramètre text
vardef long_texte(text t)=
  save i;
  numeric i;
  i:=0;
  for n=t: i:=i+1; endfor
  i
enddef;


% Placement fin des étiquettes
vardef thelabelang@#(expr s,z)=
     save tmpang,tmpfig,tmppt,tmppath,tmpstr;
     string tmpstr;
     numeric tmpang;
     pair tmppt;
     path tmppath;
     save p; picture p;
     tmpstr := str @#;
     if picture s:  p=s
     else:    p = textext(s)
     fi;
     tmppath := llcorner p -- lrcorner p -- urcorner p -- ulcorner p -- cycle;
     if ASCII tmpstr = 91:
        tmpang := scantokens substring(1,length(tmpstr)-1) of tmpstr
     else:
        tmpang := scantokens tmpstr
     fi;
     tmppt := tmppath intersectionpoint ((0.5urcorner p+0.5llcorner p)-- (0.5urcorner p +0.5llcorner p+ 100 * (dir (180+tmpang))));
      p shifted (-tmppt + z + labeloffset*(cosd(tmpang),sind(tmpang)) )
enddef;


% draw et fill sans _op_
def myfill expr c = addto currentpicture contour c enddef;
def mydraw expr p =
  addto currentpicture
  if picture p:
    also p
  else:
    doublepath p withpen currentpen
  fi
enddef;




def labelang = draw thelabelang enddef;
def labelmp = mydraw thelabel enddef;


vardef ptofpath@#(expr p, c)= % Trouve le point du chemin p à la position @# par rapport à c
 save diam; numeric diam; diam:=abs(urcorner bbox p - llcorner bbox p);
 save tmpang;numeric tmpang;
 if (str @#=""):
   c
 elseif ((ASCII str @# < 58) or (ASCII str @# = 91)):
   if ASCII str @# = 91:
      tmpang := scantokens substring(1,length(str @#)-1) of str @#
   else:
      tmpang := scantokens str @#
   fi;
   (c--((c+diam*(1,0)) rotatedaround(c,tmpang))) intersectionpoint p
 else:
  (c--(c+diam*laboff@#)) intersectionpoint p
 fi
enddef;


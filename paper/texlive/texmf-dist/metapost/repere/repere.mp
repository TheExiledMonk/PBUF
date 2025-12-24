%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                        repere.mp                           %%
%%   Macros pour la construction de figures dans un repère    %%
%%                    o.peault@posteo.net                     %%
%%                Version 23.05.b (Mai 2023)                  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This work may be distributed and/or modified under the conditions of
% the LaTeX Project Public License, either version 1.3c of this license
% or (at your option) any later version.  The latest version of this
% license is in http://www.latex-project.org/lppl.txt
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Mars 2019 : changement spline
% Juin 2019 : demi-droites, nomme[<angle>]
% Mars 2020 : filldraw
% Octobre 2021 : nommerot, cote
% Juin 2022 : marquesegment plusieurs segments
%             angle_cote
%             segments, vecteurs, polygones complets
% Octobre-Décembre 2022 : doc
% Février 2023 : Main levée
% Mai 2023 : Transformations, couleur de fond, pixart

input format;
if not known mplib: input latexmp fi;

warningcheck:=0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   REPERE   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numeric _nfig,_nbaxes,theta,taillegrad,taillepoint,taillecroix,facteurx,facteury;
boolean flecheaxe,displayfrac,coupe,gradxpart,gradypart;

numeric Xmin,Xmax,Ux,Ymin,Ymax,Uy,theta,Uxy,Lx,Ly;
Xmin:=-10;Xmax:=10;Ux:=1cm;
Ymin:=-10;Ymax:=10;Uy:=1cm;
Lx:=10cm;Ly:=10cm;
theta:=90;
Uxy:=-1;


numeric xminf,xmaxf,nbf;
% pour les tracés de fonctions

_nfig:=1;
path cadre;
numeric _diag,_largcadre,_hautcadre,_tfig;
pair _O,_I,_J,_O_;
transform _T;
picture _axorlab; pair poslabO;

string marque_p,fraction;

facteury:=1;

boolean rep_groupe;
rep_groupe:=false;

%%%%%%%%%%% Sauvegarde macros plain

let oldbeginfig=beginfig;
let oldendfig=endfig;

let olddraw = draw;
let olddrawarrow = drawarrow;
let olddrawdblarrow = drawdblarrow;
let oldfill = fill;
let oldfilldraw = filldraw;
vardef theoldlabel@#(expr s,z) =  % Position s near z
  save p; picture p;
  if picture s:  p=s
  else:    p = s infont defaultfont scaled defaultscale
  fi;
  p shifted (z + labeloffset*laboff@# -
     (labxf@#*lrcorner p + labyf@#*ulcorner p
       + (1-labxf@#-labyf@#)*llcorner p
     )
  )
enddef;
vardef oldbbox primary p =
  llcorner p-(bboxmargin,bboxmargin) -- lrcorner p+(bboxmargin,-bboxmargin)
  -- urcorner p+(bboxmargin,bboxmargin) -- ulcorner p+(-bboxmargin,bboxmargin)
  -- cycle
enddef;


%%%%%%% Redéfinition beginfig

def beginfig(expr c)=
  oldbeginfig(c);
  _nfig:=c;
  rep_groupe:=true
enddef;

def endfig=
  oldendfig;
  _nfig:=_nfig+1;
  rep_groupe:=false
enddef;


%%%%%%%%%% Définition du repère

vardef repere@#(text t)=
 if (str @#="") or (str @#="larg") or (str @#="width") or (str @#="orth") or (str @#="prop"): 
    definitionrepere@#(t)
 else:
    reperenum@#(t)
 fi
enddef;


vardef reperenum[]@#(text t)=
  _nfig:=@;
  definitionrepere@#(t)
enddef;

boolean bf;
bf:=false;

vardef definitionrepere@#(text t) =
  save i,_def;
  if not rep_groupe: oldbeginfig(_nfig);bf:=true fi;
  numeric _def[],i,unitex,unitey,unite;
  _def[7]:=90;i:=1;
  _def1:=Xmin; _def2:=Xmax; _def3:=Ux;
  _def4:=Ymin; _def5:=Ymax; _def6:=Uy;
  _def7:=theta;
  if Uxy>0: _def3:=Uxy;_def6:=Uxy fi;
  if (str @#="larg") or (str @#="width"): _def3:=Lx;_def6:=Ly fi;
  for _i=t: _def[i]:=_i; i:=i+1; endfor;
  Xmin:=_def1; Xmax:=_def2; Ymin:=_def4; Ymax:=_def5; theta:=_def7;
  if (str @#="larg") or (str @#="width"): _def3:=(_def3-_def6*cosd(theta)/sind(theta))/(Xmax-Xmin);_def6:=_def6/((Ymax-Ymin)*sind(theta))
  elseif str @#="orth": _def3:=_def3/(Xmax-Xmin); _def6:=_def3;theta:=90 
  elseif str @#="prop": _def3:=_def3/(Xmax-Xmin); _def6:=_def6*_def3;theta:=90
  fi;
  Ux:=_def3;Uy:=_def6;
  _T:=identity xscaled Ux yscaled Uy slanted cosd(theta) yscaled sind(theta);
  _defcadre;
  settout(Xmin,Xmax,Ymin,Ymax);
  _O:=_cart(0,0); xO:=0 ; yO:=0;_O_:=(0,0);
  _I:=_cart(1,0); _J:=_cart(0,1);
  taillegrad:=0.8mm;taillepoint:=3;taillecroix:=5;_nbaxes:=0;_axorlab:=nullpicture;poslabO:=(0,0);
  marque_p := "*";displayfrac:=false;flecheaxe:=true;coupe:=true;
  gradxpart:=true;gradypart:=true;
  defaultscale:=0.9;
  unitex:=abs(_I);unitey:=abs(_J);unite:=unitex;
  facteurx:=1;facteury:=1;
  string extranumx,extranumy;
  extranumx=extranumy="";
     xminf:=Xmin;xmaxf:=Xmax;nbf:=50; % défaut tracés fonctions
  init_traces;
enddef;

%%%% Initialisations tracés (redéfinition draw etc.)
def init_traces =
  %% draw
  def ddraw expr p =
    addto currentpicture
    if picture p:
      also p
    else:
      doublepath p transformed _T withpen currentpen
    fi
    _op_
  enddef;
  def Draw expr p=
    if (path p) and (type_trace="mainlevee"):
         ddraw mainleveepath(p)
    else:
         ddraw p
    fi
  enddef;
  def draw expr p= Draw p enddef;
  %% fill
  def fill expr c =
    if path c:
       addto currentpicture contour (c transformed _T) _op_
    else:
      olddraw c
    fi
  enddef;
  %% fill
  def filldraw expr c =
  addto currentpicture contour (c transformed _T) withpen currentpen
  _op_ enddef;
  %% thelabel
  vardef thelabel@#(expr s,z) =  % Position s near z
    save p; picture p;
    if picture s:  p=s
    elseif path s: p=image(draw s)
    else:          p = LaTeX(s)
    fi;
    p shifted (_cart(z) + labeloffset*laboff@# -
       (labxf@#*lrcorner p + labyf@#*ulcorner p
         + (1-labxf@#-labyf@#)*llcorner p
       )
    )
  enddef;
  vardef thelabelrot@#(expr s,z,an) =  % Position s near z
    save p; picture p;
    if picture s:  p=s
    else:    p = LaTeX(s)
    fi;
    p:=p rotatedaround (center bbox p,an);
    p shifted (_cart(z) + labeloffset*laboff@# -
       (labxf@#*lrcorner p + labyf@#*ulcorner p
         + (1-labxf@#-labyf@#)*llcorner p
       )
    )
  enddef;
  def labelrot = draw thelabelrot enddef;
  %% drawarrow
  def ddrawarrow expr p = _apth:=p transformed _T; _finarr enddef;
  def drawarrow expr p=
    if type_trace="mainlevee":
         ddrawarrow mainleveepath(p)
    else:
       ddrawarrow p
    fi
  enddef;
  def ddrawdblarrow expr p = _apth:=p transformed _T; _findarr enddef;
  def drawdblarrow expr p=
    if type_trace="mainlevee":
         ddrawdblarrow mainleveepath(p)
    else:
       ddrawdblarrow p
    fi
  enddef;
  %% bbox
  vardef bbox primary p =
    if picture p: oldbbox p transformed inverse _T
    else: oldbbox p
    fi
  enddef;
enddef;

def traces_orig =
  def fill expr c = addto currentpicture contour c _op_ enddef;
  def draw expr p =
    addto currentpicture
    if picture p:
      also p
    else:
      doublepath p withpen currentpen
    fi
    _op_
  enddef;
  def filldraw expr c =
    addto currentpicture contour c withpen currentpen
    _op_ enddef;
  vardef thelabel@#(expr s,z) =  % Position s near z
    save p; picture p;
    if picture s:  p=s
    else:    p = s infont defaultfont scaled defaultscale
    fi;
    p shifted (z + labeloffset*laboff@# -
       (labxf@#*lrcorner p + labyf@#*ulcorner p
         + (1-labxf@#-labyf@#)*llcorner p
       )
    )
  enddef;
  vardef bbox primary p =
    llcorner p-(bboxmargin,bboxmargin) -- lrcorner p+(bboxmargin,-bboxmargin)
    -- urcorner p+(bboxmargin,bboxmargin) -- ulcorner p+(-bboxmargin,bboxmargin)-- cycle
  enddef;
  def drawarrow expr p = _apth:=p; _finarr enddef;
  def drawdblarrow expr p = _apth:=p; _findarr enddef;

enddef;

def _finarr text t =
  olddraw _apth t;
  oldfilldraw arrowhead _apth  t
enddef;

def _findarr text t =
  olddraw _apth t;
  oldfilldraw arrowhead _apth withpen currentpen  t;
  oldfilldraw arrowhead  reverse _apth  withpen currentpen  t
enddef;

vardef arrowhead expr p = %redéfinition flèches
  save q,e; path q; pair e;
  e = point length p of p;
  q = gobble(p shifted -e cutafter makepath(pencircle scaled 2ahlength))
    cuttings;
  (q rotated .5ahangle & reverse q rotated -.5ahangle -- 0.5*(point 0 of q)-- cycle)  shifted e
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Main levée %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

vardef signe(expr n)=
  if n=0:
    0
  elseif n>0:
    1
  else:
    -1
  fi
enddef;


% D'après Toby Thurston :  https://raw.githubusercontent.com/thruston/Drawing-with-Metapost/master/Drawing-with-Metapost.pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%

string type_trace;
type_trace:="";

def anglemainlevee(text t) =
  def _anglemainlevee_ = t enddef
enddef;

anglemainlevee(5*signe(normaldeviate)+normaldeviate);


%% Chemin de longueur 1, on rajoute l'angle aux extrémités
def mainleveeseg(expr p) =
  point 0 of p {(direction 0 of p) rotated (_anglemainlevee_)} ..
  point 1 of p {(direction 1 of p) rotated (_anglemainlevee_)}
enddef;

%% Chemin de longueur quelconque, on divise en chemins simples
def mainleveepath(expr p) =
    mainleveeseg(subpath(0,1) of p)
       for i=1 upto length(p)-1:
           & mainleveeseg(subpath(i,i+1) of p)
       endfor
    if cycle p: & cycle fi
enddef;
string _tmp_type_trace_;
_tmp_type_trace_:="";

def drawmainlevee text t=
  _tmp_type_trace_:=type_trace;
  type_trace:="mainlevee";
  draw t;
  type_trace:=_tmp_type_trace_;
enddef;

def mainlevee(text t)=
 if path t:
    mainleveepath(t)
 else:
   image(drawmainlevee t)
 fi
enddef;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def setaxes(expr a,b,c,d)=
  AxeXmin:=a;AxeXmax:=b;AxeYmin:=c;AxeYmax:=d;
  setgrad(a,b,c,d);
  setval(a,b,c,d);
enddef;

def setgrad(expr a,b,c,d)=
  GradXmin:=a;GradXmax:=b;GradYmin:=c;GradYmax:=d
enddef;

def setval(expr a,b,c,d)=
  ValXmin:=a;ValXmax:=b;ValYmin:=c;ValYmax:=d
enddef;

def setquad(expr a,b,c,d)=
  QuadXmin:=a;QuadXmax:=b;QuadYmin:=c;QuadYmax:=d
enddef;

def setrepere(expr a,b,c,d)=
  setaxes(a,b,c,d);
  setgrad(a,b,c,d);
  setval(a,b,c,d);
  setquad(a,b,c,d);
enddef;

let setall=setrepere;
let settout=setrepere;

color fond_coul;


def fin =
  if _nbaxes=1: olddraw _axorlab fi;
  if coupe: clip currentpicture to (cadre transformed _T) fi;
%  Gestion du fond
  if known fond_coul:
    background:=fond_coul;
    save tmpfig;
    picture tmpfig;
    tmpfig:=currentpicture;
    currentpicture:=nullpicture;
    newinternal bboxmargin;
    bboxmargin:=0bp;
    fill bbox tmpfig withcolor background;
    draw tmpfig;
  fi;
% Fin gestion du fond
  if bf: oldendfig; bf:=false;_nfig:=_nfig+1 fi;
  rep_tab:=false;   % Ajout octobre 2021
  traces_orig;
enddef;

def _defcadre=
   cadre:=(Xmin,Ymin)--(Xmin,Ymax)--(Xmax,Ymax)--(Xmax,Ymin)--cycle;
   XmiN:=Xmin-0.05(Xmax-Xmin);XmaX:=Xmax+0.05(Xmax-Xmin);
   YmiN:=Ymin-0.05(Ymax-Ymin);YmaX:=Ymax+0.05(Ymax-Ymin);
   _diag:=abs(llcorner cadre - urcorner cadre);
   _diag_:=abs(llcorner _cart(cadre) - urcorner _cart(cadre));
   _largcadre:=abs(llcorner cadre - lrcorner cadre);
   _largcadre_:=abs(llcorner _cart(cadre) - lrcorner _cart(cadre));
   _hautcadre:=abs(llcorner cadre - ulcorner cadre);
   _hautcadre_:=abs(llcorner _cart(cadre) - ulcorner _cart(cadre));
   _tfig:=max(_largcadre,_hautcadre);         % compatibilité geometriesyr
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   ORIGINE   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def interaxes(expr a,b) =
  _O:=_cart(a,b);_O_:=(a,b);
  xO:=a;
  yO:=b;
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   AXES   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pair _posgradx, _posgradx.lft, _posgradx.rt, _posgradx.ulft, _posgradx.top, _posgradx.urt, _posgradx.llft, _posgradx.bot, _posgradx.lrt;
_posgradx=_posgradx.bot=_posgradx.lft=_posgradx.rt=-_posgradx.top=(0,-1);
-_posgradx.llft=_posgradx.urt=(0.5,0.5);_posgradx.lrt=-_posgradx.ulft=(0.5,-0.5);


pair _posgrady, _posgrady.lft, _posgrady.rt, _posgrady.ulft, _posgrady.top, _posgrady.urt, _posgrady.llft, _posgrady.bot, _posgrady.lrt;
_posgrady=_posgrady.bot=_posgrady.lft=_posgrady.top=-_posgrady.rt=(-1,0);
-_posgrady.llft=_posgrady.urt=(0.5,0.5);_posgrady.lrt=-_posgrady.ulft=(0.5,-0.5);

path _dessin_axe_x_,_dessin_axe_y_;

vardef _etiqx_@#(expr ab,dec,st)=   %étiquettes axe des abscisses
  save pic,tmppos;
  picture pic;
   pair tmppos;
   tmppos = _dessin_axe_x_ intersectionpoint droite(ab);
  if numeric st: pic:=LaTeX("$\num{" & decimal(st) & extranumx & "}$")
  elseif string st: pic:=LaTeX(st)
  else: pic:=st fi;
%  theoldlabel.@#(pic,_cart(ab,yO) shifted (dec*_posgradx@#))
  theoldlabel.@#(pic,_cart(tmppos) shifted (dec*_posgradx@#))
enddef;

vardef _etiqy_@#(expr ro,dec,st)=   %étiquettes axe des ordonnées
  save pic,tmppos;
  picture pic;
   pair tmppos;
   tmppos = _dessin_axe_y_ intersectionpoint droite(0,ro);
  if numeric st: pic:=LaTeX("$\num{" & decimal(st) & extranumy & "}$")
  elseif string st: pic:=LaTeX(st)
  else: pic:=st fi;
%  theoldlabel.@#(pic,_cart(xO,ro) shifted (dec*_posgrady@#))
  theoldlabel.@#(pic,_cart(tmppos) shifted (dec*_posgrady@#))
enddef;


vardef graduationx(expr x)=          %%%%%%% graduation axe des abscisses
   save tmppos;
   pair tmppos;
   tmppos = _dessin_axe_x_ intersectionpoint droite(x);
%   ((0,taillegrad)--(0,-taillegrad)) rotated (theta-90) shifted _cart(x,yO)
   ((0,taillegrad)--(0,-taillegrad)) rotated (theta-90) shifted _cart(tmppos)
enddef;


vardef graduationy(expr y)=          %%%%%%% graduation axe des ordonnées
   save tmppos;
   pair tmppos;
   tmppos = _dessin_axe_y_ intersectionpoint droite(0,y);
%   ((taillegrad,0)--(-taillegrad,0)) shifted _cart(xO,y)
   ((taillegrad,0)--(-taillegrad,0)) shifted _cart(tmppos)
enddef;

vardef etiquettex@#(expr x)=
   _etiqx_@#(x,taillegrad,x*facteurx)
enddef;


vardef etiquettey@#(expr y)=
   _etiqy_@#(y,taillegrad,y*facteury)
enddef;


%vardef dessinaxex=_cart(AxeXmin,yO)--_cart(AxeXmax,yO) enddef;

%vardef dessinaxey=_cart(xO,AxeYmin)--_cart(xO,AxeYmax) enddef;

vardef cheminaxex=
      (AxeXmin,yO)--
        if (AxeXmin<xO) and (xO<AxeXmax):
           (xO,yO)--
        fi
      (AxeXmax,yO)
enddef;

vardef cheminaxey=
      (xO,AxeYmin)--
        if (AxeYmin<yO) and (yO<AxeYmax):
           (xO,yO)--
        fi
      (xO,AxeYmax)
enddef;


vardef graduationsaxex(expr grad)=      %%%% ensemble des graduations abscisses
   save $;picture $;
   $=image(%
    imax:=floor((GradXmax-xO)/grad);imin:=ceiling((GradXmin-xO)/grad);
    for i=imin upto imax-1:
     olddraw graduationx(arrondimil(i*grad)+xO) withpen pencircle scaled 0.5bp;
    endfor
    % dernière graduation
    if AxeXmax*Ux-(imax*grad+xO)*Ux>ahlength: 
     olddraw graduationx(arrondimil(imax*grad)+xO) withpen pencircle scaled 0.5bp
    fi);
   $
enddef;

vardef graduationsaxey(expr grad)=      %%%% ensemble des graduations ordonnées
   save $;picture $;
   $=image(%
    imax:=floor((GradYmax-yO)/grad);imin:=ceiling((GradYmin-yO)/grad);
    for i=imin upto imax-1:
     olddraw graduationy(arrondimil(i*grad)+yO) withpen pencircle scaled 0.5bp;
    endfor;
    % dernière graduation
    if AxeYmax*Uy-(imax*grad+yO)*Uy>ahlength: olddraw graduationy(arrondimil(imax*grad)+yO) withpen pencircle scaled 0.5bp fi);
   $
enddef;


vardef etiquettesaxex@#(expr val)=       %%% ensemble des étiquettes abscisses
   save $,imin,imax,labmin,labmax;
   picture $,labmin,labmax;
   numeric imin,imax;
     $=image(%
   	imin:=ceiling((ValXmin-xO)/val);imax:=floor((ValXmax-xO)/val);
   	% définition première étiquette
        if (imin<>0):
   	    labmin:= etiquettex.@#(arrondimil(imin*val)+xO)
	   else: labmin:=nullpicture
   	fi;
   	% définition dernière étiquette
        if (imax<>0):
   	    labmax:= etiquettex.@#(arrondimil(imax*val)+xO)
	   else: labmax:=nullpicture
   	fi;
   	% tracé première étiquette
	   if xpart ((ulcorner labmin) transformed inverse _T)>=Xmin: olddraw labmin fi;
   	% tracé autres étiquettes
   	for i=imin+1 upto imax-1:
    		if (i<>0):
    		  olddraw etiquettex.@#(arrondimil(i*val)+xO)
    		fi;
   	endfor;
   	% tracé dernière étiquette
   	if xpart ((lrcorner labmax) transformed inverse _T)<=Xmax: olddraw labmax fi;
        );
      $
enddef;

vardef etiquettesaxey@#(expr val)=       %%% ensemble des étiquettes ordonnées
   save $,imin,imax,labmin,labmax;
   picture $,labmin,labmax;
   numeric imin,imax;
     $=image(%
   	imin:=ceiling((ValYmin-yO)/val);imax:=floor((ValYmax-yO)/val);
   	% définition première étiquette
        if (imin<>0):
   	    labmin:= etiquettey.@#(arrondimil(imin*val)+yO)
	   else: labmin:=nullpicture
   	fi;
   	% définition dernière étiquette
        if (imax<>0):
   	    labmax:= etiquettey.@#(arrondimil(imax*val)+yO)
	   else: labmax:=nullpicture
   	fi;
   	% tracé première étiquette
   	if ypart (llcorner labmin) >=ypart _cart(Xmin,Ymin): olddraw labmin fi;
   	% tracé autres étiquettes
   	for i=imin+1 upto imax-1:
    		if (i<>0):
    		  olddraw etiquettey.@#(arrondimil(i*val)+yO)
    		fi;
   	endfor;
   	% tracé dernière étiquette
   	if ypart (ulcorner labmax) <=ypart _cart(Xmax,Ymax): olddraw labmax fi
         );
      $
enddef;

vardef _axex_@#(expr grad,val,nn)=
   save $;
   picture $;
   _dessin_axe_x_:= 
              if type_trace="mainlevee": 
                  mainlevee(cheminaxex) 
              else: 
                  cheminaxex
              fi;
   $=image(
   %    axe
   if flecheaxe: ddrawarrow  else: ddraw fi _dessin_axe_x_;
   %    graduations
   if grad>0: olddraw graduationsaxex(grad) fi;
   %    étiquettes
   if val>0:
   	if nn=1:
	       if str @#<>"":
		  olddraw etiquettesaxex@#(val);
		  _axorlab:=etiquettex@#(xO)
	       else:
		  olddraw etiquettesaxex.bot(val);
		  _axorlab:=etiquettex.bot(xO)
	       fi;
	       poslabO:=poslabO + unitvector(0,ypart _posgradx@#);
	       if xO<>yO:
		  olddraw _axorlab
	       elseif _nbaxes=2:
		  olddraw _etiqx_(xO,0,xO) shifted (0.7*abs(llcorner _axorlab - urcorner _axorlab)*poslabO)
	       fi
	elseif nn=2:
	       if str @#<>"":
		  olddraw etiquettesaxex@#(val);
		  olddraw etiquettex@#(xO)
	       else:
		  olddraw etiquettesaxex.bot(val);
		  olddraw etiquettex.bot(xO)
	       fi
	else:
	       if str @#<>"":
		  olddraw etiquettesaxex@#(val)
	       else:
		  olddraw etiquettesaxex.bot(val)
	       fi
	fi
   fi
   );
   $
enddef;

vardef _axey_@#(expr grad,val,nn)=
   save $;
   picture $;
   _dessin_axe_y_:=
        if type_trace="mainlevee":
           mainlevee(cheminaxey)
        else:
           cheminaxey
        fi;
   $=image(
   %    axe
   if flecheaxe: ddrawarrow  else: ddraw fi _dessin_axe_y_;
   %    graduations
   if grad>0: olddraw graduationsaxey(grad) fi;
   %    étiquettes
   if val>0:
   	if nn=1:
	       if str @#<>"":
		  olddraw etiquettesaxey@#(val);
		  _axorlab:=etiquettey@#(yO)
	       else:
		  olddraw etiquettesaxey.lft(val);
		  _axorlab:=etiquettey.lft(yO)
	       fi;
	       poslabO:=poslabO + unitvector(xpart _posgrady@#,0);
	       if xO<>yO:
		  olddraw _axorlab
	       elseif _nbaxes=2:
		  olddraw _etiqy_(yO,0,yO) shifted (0.7*abs(llcorner _axorlab - urcorner _axorlab)*poslabO)
	       fi
	elseif nn=2:
	       if str @#<>"":
		  olddraw etiquettesaxey@#(val);
		  olddraw etiquettey@#(yO)
	       else:
		  olddraw etiquettesaxey.lft(val);
		  olddraw etiquettey.lft(yO)
	       fi
	else:
	       if str @#<>"":
		  olddraw etiquettesaxey@#(val)
	       else:
		  olddraw etiquettesaxey.lft(val)
	       fi
	fi
   fi
   );
   $
enddef;

vardef axex@# (expr grad,val)=  %%%%%%axe des abscisses avec gestion graduation origine
   _nbaxes:=_nbaxes+1;
   _axex_@#(grad,val,1)
enddef;

vardef axey@# (expr grad,val)=  %%%%%%axe des ordonnées avec gestion graduation origine
   _nbaxes:=_nbaxes+1;
   _axey_@#(grad,val,1)
enddef;

vardef axexo@#(expr grad,val)=  %%%%%%axe des abscisses avec graduation origine
   _axex_@#(grad,val,2)
enddef;

vardef axeyo@#(expr grad,val)=  %%%%%%axe des ordonnées avec graduation origine
   _axey_@#(grad,val,2)
enddef;

vardef axexn@#(expr grad,val)=  %%%%%%axe des abscisses sans graduation origine
   _axex_@#(grad,val,3)
enddef;

vardef axeyn@#(expr grad,val)=  %%%%%%axe des ordonnées sans graduation origine
   _axey_@#(grad,val,3)
enddef;

vardef axes@# (expr grad,val)= %%%% les deux axes
   save $,posx,posy;
   picture $;
   numeric posx,posy;
   posx:=ypart (_posgradx@#+_posgrady@#);
   posy:=xpart (_posgradx@#+_posgrady@#);
   $=image(%
   if posx>0: draw axex.top(grad,val) else: draw axex.bot(grad,val) fi;
   if posy>0: draw axey.rt(grad,val) else: draw axey.lft(grad,val) fi
   );
   $
enddef;

vardef axeso@# (expr grad,val)= %%%% les deux axes avec origine
   save $,posx,posy;
   picture $;
   numeric posx,posy;
   posx:=ypart (_posgradx@#+_posgrady@#);
   posy:=xpart (_posgradx@#+_posgrady@#);
   $=image(%
   if posx>0: draw axexo.top(grad,val) else: draw axexo.bot(grad,val) fi;
   if posy>0: draw axeyo.rt(grad,val) else: draw axeyo.lft(grad,val) fi
   );
   $
enddef;

vardef axesn@# (expr grad,val)= %%%% les deux axes sans origine
   save $,posx,posy;
   picture $;
   numeric posx,posy;
   posx:=ypart (_posgradx@#+_posgrady@#);
   posy:=xpart (_posgradx@#+_posgrady@#);
   $=image(%
   if posx>0: draw axexn.top(grad,val) else: draw axexn.bot(grad,val) fi;
   if posy>0: draw axeyn.rt(grad,val) else: draw axeyn.lft(grad,val) fi
   );
   $
enddef;

% axes gradués avec multiples de pi

vardef fracmulttext(expr n,d,tt)=  %%%% construction de la fraction n*tt/d
   save nn,dd,s,fr,$,fraction;
   string s,fr,fraction;
   picture $;
   numeric nn,dd;
   nn:=abs((n)/pgcd(n,d));dd:=d/pgcd(n,d);
   if displayfrac:
     fraction:="d"
     else: fraction:=""
   fi;
   if dd=1:
   	if nn=1:
   	  fr:=tt
   	  else: fr:=decimal nn & tt
   	fi
   else:
   	if nn=1:
   	  fr:="\" & fraction & "frac{" & tt & "}{" & decimal dd & "}"
   	  else: fr:="\" & fraction & "frac{" & decimal nn & tt & "}{" & decimal dd & "}"
   	fi
   fi;
   if n<0:
     s:="-"
     else: s:="";
           if n=0:
             fr:= "0"
           fi
    fi;
   $=LaTeX("$" & s & fr & "$");
   $
enddef;

vardef fracmultpi(expr n,d)=
  fracmulttext(n,d,"\pi")
enddef;


vardef etiquettexpi@#(expr n,d)=
   _etiqx_@#(n*pi/d,taillegrad,fracmultpi(n,d))
enddef;

vardef etiquetteypi@#(expr n,d)=
   _etiqy_@#(n*pi/d,taillegrad,fracmultpi(n,d))
enddef;


vardef etiquettesaxexpi@#(expr n,d)=       %%% ensemble des étiquettes multiples de pi abscisses
   save $,imin,imax,labmin,labmax,val;
   picture $,labmin,labmax;
   numeric imin,imax,val;
    val:=n*pi/d;
     $=image(%
   	imin:=ceiling((ValXmin-xO)/val);imax:=floor((ValXmax-xO)/val);
   	% définition première étiquette
        if (imin<>0):
   	    labmin:= etiquettexpi.@#(imin*n,d)
	   else: labmin:=nullpicture
   	fi;
   	% définition dernière étiquette
        if (imax<>0):
   	    labmax:= etiquettexpi.@#(imax*n,d)
	   else: labmax:=nullpicture
   	fi;
   	% tracé première étiquette
	   if xpart ((ulcorner labmin) transformed inverse _T)>=Xmin: draw labmin fi;
   	% tracé autres étiquettes
   	for i=imin+1 upto imax-1:
    		if (i<>0):
    		  draw etiquettexpi.@#(i*n,d)
    		fi;
   	endfor;
   	% tracé dernière étiquette
   	if xpart ((lrcorner labmax) transformed inverse _T)<=Xmax: draw labmax fi;
        );
      $
enddef;

vardef etiquettesaxeypi@#(expr n,d)=       %%% ensemble des étiquettes multiples de pi ordonnées
   save $,imin,imax,labmin,labmax,val;
   picture $,labmin,labmax;
   numeric imin,imax,val;
    val:=n*pi/d;
     $=image(%
   	imin:=ceiling((ValYmin-yO)/val);imax:=floor((ValYmax-yO)/val);
   	% définition première étiquette
        if (imin<>0):
   	    labmin:= etiquetteypi.@#(imin*n,d)
	   else: labmin:=nullpicture
   	fi;
   	% définition dernière étiquette
        if (imax<>0):
   	    labmax:= etiquetteypi.@#(imax*n,d)
	   else: labmax:=nullpicture
   	fi;
   	% tracé première étiquette
   	if ypart (llcorner labmin) >=ypart _cart(Xmin,Ymin): draw labmin fi;
   	% tracé autres étiquettes
   	for i=imin+1 upto imax-1:
    		if (i<>0):
    		  draw etiquetteypi.@#(i*n,d)
    		fi;
   	endfor;
   	% tracé dernière étiquette
   	if ypart (ulcorner labmax) <=ypart _cart(Xmax,Ymax): draw labmax fi
         );
      $
enddef;

vardef _axexpi_@#(expr num,den,nn)=
   save $,grad;
   numeric grad;
   picture $;
   grad:=num*pi/den;
   _dessin_axe_x_:= 
              if type_trace="mainlevee": 
                  mainlevee(cheminaxex) 
              else: 
                  cheminaxex
              fi;
   $=image(
   %    axe
   if flecheaxe: ddrawarrow  else: ddraw fi _dessin_axe_x_;
   %    graduations
   if grad>0: draw graduationsaxex(grad) fi;
   %    étiquettes
   if nn=1:
       if str @#<>"":
	  draw etiquettesaxexpi@#(num,den);
	  _axorlab:=etiquettexpi@#(xO/pi,1)
       else:
	  draw etiquettesaxexpi.bot(num,den);
	  _axorlab:=etiquettexpi.bot(xO/pi,1)
       fi;
       poslabO:=poslabO + unitvector(0,ypart _posgradx@#);
       if xO<>yO:
	  draw _axorlab
       elseif _nbaxes=2:
	  draw  _etiqx_(xO,0,fracmultpi(xO/pi,1)) shifted (0.7*abs(llcorner _axorlab - urcorner _axorlab)*poslabO)
       fi
   elseif nn=2:
       if str @#<>"":
	  draw etiquettesaxexpi@#(num,den);
	  draw etiquettexpi@#(xO/pi,1)
       else:
	  draw etiquettesaxexpi.bot(num,den);
	  draw etiquettexpi.bot(xO/pi,1)
       fi
   else:
       if str @#<>"":
	  draw etiquettesaxexpi@#(num,den)
       else:
	  draw etiquettesaxexpi.bot(num,den)
       fi
   fi
   );
   $
enddef;

vardef _axeypi_@#(expr num,den,nn)=
   save $,grad;
   numeric grad;
   picture $;
   grad:=num*pi/den;
   _dessin_axe_y_:= 
              if type_trace="mainlevee": 
                  mainlevee(cheminaxey) 
              else: 
                  cheminaxey
              fi;
   $=image(
   %    axe
   if flecheaxe: ddrawarrow  else: ddraw fi _dessin_axe_y_;
   %    graduations
   if grad>0: draw graduationsaxey(grad) fi;
   %    étiquettes
   if nn=1:
       if str @#<>"":
	  draw etiquettesaxeypi@#(num,den);
	  _axorlab:=etiquetteypi@#(yO/pi,1)
       else:
	  draw etiquettesaxeypi.lft(num,den);
	  _axorlab:=etiquetteypi.lft(yO/pi,1)
       fi;
       poslabO:=poslabO + unitvector(xpart _posgrady@#,0);
       if xO<>yO:
	  draw _axorlab
       elseif _nbaxes=2:
	  draw  _etiqy_(yO,0,fracmultpi(yO/pi,1)) shifted (0.7*abs(llcorner _axorlab - urcorner _axorlab)*poslabO)
%	  draw _etiqy_(yO,0,yO) shifted (0.7*abs(llcorner _axorlab - urcorner _axorlab)*poslabO)
       fi
   elseif nn=2:
       if str @#<>"":
	  draw etiquettesaxeypi@#(num,den);
	  draw etiquetteypi@#(yO/pi,1)
       else:
	  draw etiquettesaxeypi.lft(num,den);
	  draw etiquetteypi.lft(yO/pi,1)
       fi
   else:
       if str @#<>"":
	  draw etiquettesaxeypi@#(num,den)
       else:
	  draw etiquettesaxeypi.lft(num,den)
       fi
   fi
   );
   $
enddef;

vardef axexpi@# (expr n,d)=  %%%%%%axe des abscisses avec gestion graduation origine mult pi
   _nbaxes:=_nbaxes+1;
   _axexpi_@#(n,d,1)
enddef;

vardef axeypi@# (expr n,d)=  %%%%%%axe des ordonnées avec gestion graduation origine mult pi
   _nbaxes:=_nbaxes+1;
   _axeypi_@#(n,d,1)
enddef;

vardef axexpio@#(expr n,d)=  %%%%%%axe des abscisses avec graduation origine mult pi
   _axexpi_@#(n,d,2)
enddef;

vardef axeypio@#(expr n,d)=  %%%%%%axe des ordonnées avec graduation origine mult pi
   _axeypi_@#(n,d,2)
enddef;

vardef axexpin@#(expr n,d)=  %%%%%%axe des abscisses sans graduation origine mult pi
   _axexpi_@#(n,d,3)
enddef;

vardef axeypin@#(expr grad,val)=  %%%%%%axe des ordonnées sans graduation origine mult pi
   _axeypi_@#(n,d,3)
enddef;

vardef axespi@# (expr n,d)= %%%% les deux axes mult pi
   save $,posx,posy;
   picture $;
   numeric posx,posy;
   posx:=ypart (_posgradx@#+_posgrady@#);
   posy:=xpart (_posgradx@#+_posgrady@#);
   $=image(%
   if posx>0: draw axexpi.top(n,d) else: draw axexpi.bot(n,d) fi;
   if posy>0: draw axeypi.rt(n,d) else: draw axeypi.lft(n,d) fi
   );
   $
enddef;

vardef axespio@# (expr n,d)= %%%% les deux axes avec origine mult pi
   save $,posx,posy;
   picture $;
   numeric posx,posy;
   posx:=ypart (_posgradx@#+_posgrady@#);
   posy:=xpart (_posgradx@#+_posgrady@#);
   $=image(%
   if posx>0: draw axexpio.top(n,d) else: draw axexpio.bot(n,d) fi;
   if posy>0: draw axeypio.rt(n,d) else: draw axeypio.lft(n,d) fi
   );
   $
enddef;

vardef axespin@# (expr n,d)= %%%% les deux axes sans origine mult pi
   save $,posx,posy;
   picture $;
   numeric posx,posy;
   posx:=ypart (_posgradx@#+_posgrady@#);
   posy:=xpart (_posgradx@#+_posgrady@#);
   $=image(%
   if posx>0: draw axexpin.top(n,d) else: draw axexpin.bot(n,d) fi;
   if posy>0: draw axeypin.rt(n,d) else: draw axeypin.lft(n,d) fi
   );
   $
enddef;


%%%%%%%% graduations isolées

vardef axexpart@#(text t)=
   save $,tmp,var;
   picture $;
   numeric tmp;
   boolean var;
   var:=false;
   $=image(%
	for _i=t:
	   if numeric _i:
		if var: if str @#="": draw _etiqx_.bot(tmp,taillegrad,tmp) else: draw _etiqx_@#(tmp,taillegrad,tmp) fi fi;
		if gradxpart:
		   olddraw graduationx(_i) withpen pencircle scaled 0.8bp
		fi;
		tmp:=_i;
		var:=true
	   else:
		if str @#="": draw _etiqx_.bot(tmp,taillegrad,_i) else: draw _etiqx_@#(tmp,taillegrad,_i) fi;
		var:=false
	   fi;
	endfor;
   if var: if str @#="": draw _etiqx_.bot(tmp,taillegrad,tmp) else: draw _etiqx_@#(tmp,taillegrad,tmp) fi fi);
   $
enddef;

vardef axeypart@#(text t)=
   save $,tmp,var;
   picture $;
   numeric tmp;
   boolean var;
   var:=false;
   $=image(%
	for _i=t:
	   if numeric _i:
		if var: if str @#="": draw _etiqy_.lft(tmp,taillegrad,tmp) else: draw _etiqy_@#(tmp,taillegrad,tmp) fi fi;
%		olddraw ((taillegrad,0)--(-taillegrad,0)) shifted _cart(xO,_i) withpen pencircle scaled 0.8bp;
		if gradypart:
		   olddraw graduationy(_i) withpen pencircle scaled 0.8bp
		fi;
		tmp:=_i;
		var:=true
	   else:
		if str @#="": draw _etiqy_.lft(tmp,taillegrad,_i) else: draw _etiqy_@#(tmp,taillegrad,_i) fi;
		var:=false
	   fi;
	endfor;
   if var: if str @#="": draw _etiqy_.lft(tmp,taillegrad,tmp) else: draw _etiqy_@#(tmp,taillegrad,tmp) fi fi);
   $
enddef;


vardef axesunit=
  save $;
  picture $;
  $=image(%
        draw axes(0,0);
        draw axexpart(1);
        draw axeypart(1);
        label.llft("$0$",_cart(0,0));
          );
  $
enddef;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   POINTS - VECTEURS   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef _ccoord(expr xA,yA)=
   (xA,yA) transformed _T
enddef;

vardef _coord(expr xA,yA)=
   (xA,yA)
enddef;

vardef _cobjet(expr obj)=
   obj transformed _T
enddef;

vardef _cart(text t)=
   if long_texte(t)=2: _ccoord(t)
      else: _cobjet(t)
   fi
enddef;

vardef _c(text t)=
   if long_texte(t)=2: _coord(t)
   else: t
   fi
enddef;


vardef pol(expr r,theta)=
 (r*cos(theta),r*sin(theta))
enddef;

vardef pold(expr r,theta)=
 (r*cosd(theta),r*sind(theta))
enddef;

color coulpoint,coullabel,coullabelfonc;
coulpoint:=black;coullabel:=black;coullabelfonc:=black;


vardef marquepointFig(expr pp)=      % compatibilité geometriesyr15
 save $;
 picture $;
  if (marque_p = "plein") or (marque_p="*"):
    $:=image(oldfill fullcircle scaled taillepoint shifted _cart(pp) withcolor coulpoint);
  elseif (marque_p = "creux") or (marque_p="o"):
    $:=image(%
    %oldfill fullcircle scaled taillepoint shifted _cart(pp) withcolor background;
    olddraw fullcircle scaled taillepoint shifted _cart(pp) withcolor coulpoint);
  elseif (marque_p = "croix") or (marque_p = "+"):
  $:=image(%
    olddraw ((-taillecroix/2,0)--(taillecroix/2,0)) shifted _cart(pp) withcolor coulpoint;
    olddraw ((0,-taillecroix/2)--(0,taillecroix/2)) shifted _cart(pp) withcolor coulpoint)
  elseif (marque_p = "x"):
  $:=image(%
    olddraw ((-taillecroix/2,0)--(taillecroix/2,0)) rotated 45 shifted _cart(pp) withcolor coulpoint;
    olddraw ((0,-taillecroix/2)--(0,taillecroix/2)) rotated 45 shifted _cart(pp) withcolor coulpoint)
  else: $:=nullpicture
  fi;
  $
enddef;

def marquepoint(expr pp)=      % compatibilité geometriesyr15
  draw marquepointFig(pp)
enddef;

let MarquePoint = marquepoint;

def marquepoints(text t) text q =      %geometriesyr15
  for p_ = t: if pair p_: marquepoint(p_) q; fi endfor;
enddef;

let pointe = marquepoints;

% Juin 2019 : placement fin des étiquettes (angle donné)
vardef thelabelang@#(expr s,z)=
     save tmpang,tmpfig,tmppt,tmppath,tmpstr;
     string tmpstr;
     numeric tmpang;
     pair tmppt;
     path tmppath;
     save p; picture p;
     tmpstr := str @#;
     if picture s:  p=s
     else:    p = LaTeX(s)
     fi;
     tmppath := llcorner p -- lrcorner p -- urcorner p -- ulcorner p -- cycle;
     if ASCII tmpstr = 91:
        tmpang := scantokens substring(1,length(tmpstr)-1) of tmpstr
     else:
        tmpang := scantokens tmpstr
     fi;
     tmppt := tmppath intersectionpoint ((0.5urcorner p+0.5llcorner p)-- (0.5urcorner p +0.5llcorner p+ 100 * (dir (180+tmpang))));
      p shifted (-tmppt + _cart(z) + labeloffset*(cosd(tmpang),sind(tmpang)) )
enddef;

def labelang = draw thelabelang enddef;

vardef nommepoint@#(text t)=
   save $,ch,dess,latch;
   pair $;
   string ch,latch;
   picture dess;
   if ((ASCII str @# < 58) or (ASCII str @# = 91)) and (str @#<>""):
       def mylabel = labelang enddef
   else:
       def mylabel = label enddef
   fi;
   if long_texte(t)=1:
      dess:=image(draw marquepointFig(t);
                  ch:="$" & _chaine(t) & "$";
                  if _ext_tr_<>"": ch:=aj_extr_trsf(t,ch) fi;
                  mylabel@#(ch,t) withcolor coullabel)
   else:
      dess:=image(%
      for PP=t:
        if pair PP:
           marquepoint(PP); $:=PP
        elseif string PP:
           mylabel@#(PP,$) withcolor coullabel;
        else:
           mylabel@#(PP scaled defaultscale,$) withcolor coullabel
        fi;
      endfor)
   fi;
   dess
enddef;


% Décembre 2022 définition et nom

vardef nommedef@#(suffix suff_A)(expr ptA)=
  pair suff_A;
  suff_A:=ptA;
  nomme@#(suff_A);
enddef;



% Octobre 2021
vardef thelabelangrot@#(expr s,z,an)=
     save tmpang,tmpfig,tmppt,tmppath,tmpstr;
     string tmpstr;
     numeric tmpang;
     pair tmppt;
     path tmppath;
     save p; picture p;
     tmpstr := str @#;
     if picture s:  p=s
     else:    p = LaTeX(s)
     fi;
     tmppath := llcorner p -- lrcorner p -- urcorner p -- ulcorner p -- cycle;
     p:=p rotatedaround (center tmppath,an);
     if ASCII tmpstr = 91:
        tmpang := scantokens substring(1,length(tmpstr)-1) of tmpstr
     else:
        tmpang := scantokens tmpstr
     fi;
     tmppt := (tmppath rotatedaround (center tmppath,an)) intersectionpoint ((0.5urcorner p+0.5llcorner p)-- (0.5urcorner p +0.5llcorner p + 100 * (dir (180+tmpang))));
      p shifted (-tmppt + _cart(z) + labeloffset*(cosd(tmpang),sind(tmpang)) )
enddef;

def labelangrot = draw thelabelangrot enddef;

% Octobre 2021
vardef nommepointrot@#(text t)(expr an)=
   save $,ch,dess,latch;
   pair $;
   string ch,latch;
   picture dess;
   if (ASCII str @# < 58) or (ASCII str @# = 91):
       def mylabel = labelangrot enddef
   else:
       def mylabel = labelrot enddef
   fi;
   if long_texte(t)=1:
      dess:=image(draw marquepointFig(t);
                  ch:="$" & _chaine(t) & "$";
                  if _ext_tr_<>"": ch:=aj_extr_trsf(t,ch) fi;
                  mylabel@#(ch,t,an) withcolor coullabel)
   else:
      dess:=image(%
      for PP=t:
        if pair PP: marquepoint(PP); $:=PP
        else: if string PP: mylabel@#(PP,$,an) else: mylabel@#(PP scaled defaultscale,$,an) fi withcolor coullabel
        fi;
      endfor)
   fi;
   dess
enddef;

def nommerot = draw nommepointrot enddef;

vardef nommecourbe@#(suffix p)(text t)=
   save A,ch,dess;
   pair A;
   string ch;
   picture dess;
   if long_texte(t)=1: 
      dess:=image(ch:=_chaine(p);
                  A:=(t,fonccourbe(p)(t));
                  label@#("$" & ch & "$",A) withcolor coullabelfonc)
   else:
      dess:=image(%
        for _a=t:
          if numeric _a: A:=(_a,fonccourbe(p)(_a))
          else: if string _a: label@#(_a,A) else: label@#(_a scaled defaultscale,A) fi withcolor coullabelfonc
          fi;
        endfor)
   fi;
   dess
enddef;


vardef nommeFig@#(text t)=
 save $;
 picture $;
 if long_texte(t)=4: $:=nommeangle@#(t,0)
 elseif long_texte(t)=5: $:=nommeangle@#(t)
 else:
   for i=t:
     if pair i: 
           $:=nommepoint@#(t)
     elseif str @#="": $:=nommeautoobj(t)
     else: $:=nommecourbe@#(t)
     fi;
     exitif true;
   endfor
 fi;
   $
enddef;

def nomme = draw nommeFig enddef;


%
% Notation automatique des chemins 02/17
%
string prefnomme;
prefnomme:="right";

% Permet de savoir si un point est mieux placé relativement à prefnomme
tertiarydef A estmieuxplace B =
   if prefnomme="right": xpart A > xpart B
   elseif prefnomme="left": xpart A < xpart B
   elseif prefnomme="top": ypart A > ypart B
   elseif prefnomme="bottom": ypart A < ypart B
   else: hide(errmessage "string 'prefnomme' should be set to left, right, top or bottom")
   fi
enddef;

% Cherche le "time" du meilleur point
vardef pointnommetime(expr p)=
  save Pt, nbint, P, T,pp;
  pair Pt,P[];
  numeric nbint,pp,T[];
  intercourbestimes(T)(cadre,p);
  Pt:=_O;pp:=0;
  if unknown T[1]: T[1]:= 0; T[2] :=infinity fi;
  Pt:= point T[1] of p;
  pp:=T[1];
  nbint:=1;
  forever:
    exitif not known T[nbint+1];
    nbint:=nbint+1;
    if (point T[nbint] of p) estmieuxplace Pt:
       Pt:=point T[nbint] of p;
       pp:=T[nbint];
    fi;
  endfor
  pp
enddef;

% Cherche la "meilleure" position du label du point pp de time tt du chemin courbe
vardef position(expr pp,tt,courbe)=
  save pos,tmpdir;
  string pos;
  numeric tmpdir;
  tmpdir := angle(direction tt of courbe);
  if arrondimil((Ymax-ypart pp)/(Ymax-Ymin))=0:
     if tmpdir <0: pos:="llft"
     else: pos:="lrt"
     fi
  elseif arrondimil((Ymin-ypart pp)/(Ymax-Ymin))=0:
     if tmpdir <0: pos:="urt"
     else: pos:="ulft"
     fi
  elseif arrondimil((Xmax-xpart pp)/(Xmax-Xmin))=0:
     if tmpdir <0: pos:="llft"
     else: pos:="ulft"
     fi
  elseif arrondimil((Xmin-xpart pp)/(Xmax-Xmin))=0:
     if tmpdir <0: pos:="urt"
     else: pos:="lrt"
     fi
  elseif tt=0:
     if tmpdir<0: pos:="urt"
     else: pos := "lrt"
     fi
  else:
     if tmpdir>0: pos :="ulft"
     else: pos := "llft"
     fi
  fi;
  pos
enddef;

% Notation automatique des chemins

% Avec label automatique
vardef nommeautoobjsuf(suffix p)=
  nommeautoobjtex(p,"$" & _chaine(p) & "$")
enddef;

vardef estcadree primary f =
      (xpart urcorner f <= xpart urcorner (cadre transformed _T))
  and (ypart urcorner f <= ypart urcorner (cadre transformed _T))
  and (xpart llcorner f >= xpart llcorner (cadre transformed _T))
  and (ypart llcorner f >= ypart llcorner (cadre transformed _T))
enddef;

% Avec label donné
vardef nommeautoobjtex(expr p,leg)=
  save ptA,pos,ttmp,tmpfig;
  pair ptA;
  picture tmpfig;
  numeric ttmp;
  string pos;
  ttmp:=pointnommetime(p);
  ptA:= point ttmp of p;
  pos:=position(ptA,ttmp,p);
  for i=0 upto 120:
    tmpfig :=
        if string leg:
           scantokens("thelabel." & pos)(leg,ptA)
        else:
          scantokens("thelabel." & pos)(leg scaled defaultscale,ptA)
        fi;
    exitif estcadree tmpfig;
    ptA := point (ttmp + ((-1)**(i+1))*round((i+1)/2)*length(p)/60) of p;
  endfor
  tmpfig
enddef;


vardef nommeautoobj(text t)=
  if long_texte(t)=1: nommeautoobjsuf(t)
  else: nommeautoobjtex(t)
  fi
enddef;


% obsolète juin 2022
vardef vecteur@#(expr A)(text t)=
   save $,ch,m;
   picture $;
   pair m;
   string ch;
   $=image(%
   if long_texte(t)=1: drawarrow A--A+t;ch:=_chaine(t);label@#("$\vv{" & ijmath(ch) & "}$",A+t/2)
   else:
      for u=t:
        if pair u: drawarrow A--A+u; m:=A+u/2
        else: if string u: if u<>"": label@#("$\vv{" & u & "}$",m) fi else: label@#(u,m) fi
        fi;
      endfor
   fi);
   $
enddef;

% juin 2022
def drawvecteur(expr A,u)=
  drawarrow A--A+u
enddef;



vardef marquepointcourbefig(suffix p)(text t)=
   save $;
   picture $;
   $=image(%
   for i=t:
     marquepoint((i,fonccourbe(p)(i)));
   endfor);
   $
enddef;

def marquepointcourbe= draw marquepointcourbefig enddef;

vardef marquepointcheminfig(suffix p)(text t)=
   save $;
   picture $;
   $=image(%
   for i=t:
     marquepoint(point i of p);
   endfor);
   $
enddef;

def marquepointchemin= draw marquepointcheminfig enddef;

vardef marquecheminouvertdroitefig(expr p)=
   halfcircle shifted (0,-0.5)
              scaled (2*taillepoint)
              rotated (90+angle(direction infinity of _cart(p)))
              shifted (point infinity of _cart(p))
enddef;

def marquecheminouvertdroite= olddraw marquecheminouvertdroitefig enddef;

vardef marquecheminouvertgauchefig(expr p)=
   halfcircle shifted (0,-0.5)
              scaled (2*taillepoint)
              rotated (-90+angle(direction 0 of _cart(p)))
              shifted (point 0 of _cart(p))
enddef;

def marquecheminouvertgauche= olddraw marquecheminouvertgauchefig enddef;

def marquecheminouvert(expr p) text t=
   marquecheminouvertdroite(p) t;
   marquecheminouvertgauche(p) t
enddef;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   BASE   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef ijmath(suffix s)=
 if (substring (0,1) of s)="i": s:="\imath" & substring (1,infinity) of s fi;
 if (substring (0,1) of s)="j": s:="\jmath" & substring (1,infinity) of s fi;
 s
enddef;


color coulbase;
coulbase:=black;

vardef base(suffix a,b,c)=
 save O,I,J,$;
 string O,I,J;
 picture $;
 O:=_chaine(a); I:=_chaine(b); J:=_chaine(c);
 $:=image(%
 draw vecteur.bot(_O_,(1,0),ijmath(I)) withpen pencircle scaled 0.8 withcolor coulbase;
 draw vecteur.lft(_O_,(0,1),ijmath(J)) withpen pencircle scaled 0.8 withcolor coulbase;
 nomme.llft(_O_,"$" & O & "$") withcolor coulbase
 );
 $
enddef;


vardef basep(suffix a,b,c)=
 save O,I,J,$;
 string O,I,J;
 picture $;
 O:=_chaine(a); I:=_chaine(b); J:=_chaine(c);
 $:=image(%
 nomme.llft(_O_,"$" & O & "$") withcolor coulbase;
 draw axexpart(1,"$" & I & "$") withcolor coulbase;
 draw axeypart(1,"$" & J & "$") withcolor coulbase;
 );
 $
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   QUADRILLAGE   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  numeric q_ep; color q_coul;
  q_ep:= .3bp;                    %%%%%%%% épaisseur du trait
  q_coul := .7white;              %%%%%%%% couleur du trait
  def quadstyle= enddef;

  vardef quadrillage(expr quadx,quady) =
   save $;
   picture $;
 $=image(if quadx>0:
    for i=ceiling((QuadXmin-xO)/quadx) upto floor((QuadXmax-xO)/quadx):
     olddraw _cart(arrondimil(i*quadx)+xO,QuadYmin)--_cart(arrondimil(i*quadx)+xO,QuadYmax) quadstyle withpen pencircle scaled q_ep withcolor q_coul;
    endfor;
  fi;
  if quady>0:
    for i=ceiling((QuadYmin-yO)/quady) upto floor((QuadYmax-yO)/quady):
     olddraw _cart(QuadXmin,arrondimil(i*quady)+yO)--_cart(QuadXmax,arrondimil(i*quady)+yO) withpen pencircle scaled q_ep withcolor q_coul quadstyle;
    endfor;
  fi);
  $
 enddef;

numeric pp_ep;
color pp_coul;
pp_ep:=2;
pp_coul:=black;

  vardef papierpointe(expr quadx,quady) =
   save $;
   picture $;
 $=image(if (quadx>0) and (quady>0):
    for i=ceiling(QuadXmin/quadx) upto floor(QuadXmax/quadx):
      for j=ceiling(QuadYmin/quady) upto floor(QuadYmax/quady):
        olddraw (0,0) shifted _cart(i*quadx,j*quady) withpen pencircle scaled pp_ep withcolor pp_coul;
      endfor;
    endfor;
  fi);
  $
 enddef;

 numeric pm_epa, pm_epb, pm_epc;
 color pm_coula, pm_coulb, pm_coulc;
 pm_epa := .2bp; pm_epb := .4bp; pm_epc := .6bp;              %%%%%%%%%% épaisseurs des traits
 pm_coula := .6*white; pm_coulb := .2*white; pm_coulc := black;   %%%%%%%%%% couleurs des traits

vardef papiermillimetre =
  save $;
  picture $;
  $=image(%
  for i= ceiling(QuadXmin*Ux*10/cm) upto floor(QuadXmax*Ux*10/cm):
    if i mod 5 <> 0: olddraw (i*cm/10,QuadYmin*Uy)--(i*cm/10,QuadYmax*Uy) withpen pencircle scaled pm_epa withcolor pm_coula fi;
  endfor
  for i= ceiling(QuadYmin*Uy*10/cm) upto floor(QuadYmax*Uy*10/cm):
    if i mod 5<> 0: olddraw (QuadXmin*Ux,i*cm/10)--(QuadXmax*Ux,i*cm/10) withpen pencircle scaled pm_epa withcolor pm_coula fi;
  endfor
  %
  for i= ceiling(QuadXmin*Ux/cm)*2+1 step 2 until floor(QuadXmax*Ux/cm)*2-1:
    olddraw (i*cm/2,QuadYmin*Uy)--(i*cm/2,QuadYmax*Uy) withpen pencircle scaled pm_epb withcolor pm_coulb;
  endfor
  for i= ceiling(QuadYmin*Uy/cm)*2+1 step 2 until floor(QuadYmax*Uy/cm)*2-1:
    olddraw (QuadXmin*Ux,i*cm/2)--(QuadXmax*Ux,i*cm/2) withpen pencircle scaled pm_epb withcolor pm_coulb;
  endfor
  %
  for i= ceiling(QuadXmin*Ux/cm) upto floor(QuadXmax*Ux/cm):
    olddraw (i*cm,QuadYmin*Uy)--(i*cm,QuadYmax*Uy) withpen pencircle scaled pm_epc withcolor pm_coulc;
  endfor
  for i= ceiling(QuadYmin*Uy/cm) upto floor(QuadYmax*Uy/cm):
    olddraw (QuadXmin*Ux,i*cm)--(QuadXmax*Ux,i*cm) withpen pencircle scaled pm_epc withcolor pm_coulc;
  endfor);
  $
enddef;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%  FONCTIONS - COURBES   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vardef courbef_(suffix f)(expr xmin, xmax, n) =   %Courbe représentative d'une fonction
   save $,x;
   path $;
   $=((xmin,f(xmin))
    for i=1 upto n-1:
      ..(xmin+i*(xmax-xmin)/(n-1),f(xmin+i*(xmax-xmin)/(n-1)))
    endfor );
   $
enddef;


vardef courbef_l(suffix f)(expr xmin, xmax, n) =   %Courbe représentative d'une fonction
   save $,x;
   path $;
   $=((xmin,f(xmin))
    for i=1 upto n-1:
      ...(xmin+i*(xmax-xmin)/(n-1),f(xmin+i*(xmax-xmin)/(n-1)))
    endfor );
   $
enddef;


vardef courbefonc(suffix f)(text t)=
   save cc;path cc;
   save k,Val;numeric k,Val[];
   Val[1]:=xminf;Val[2]:=xmaxf;Val[3]:=nbf;
   k:=1;
   for i=t:
      Val[k]:=i;
      k:=k+1;
   endfor
   courbef_(f,Val[1],Val[2],Val[3])
enddef;


vardef courbefoncl(suffix f)(text t)=
        save cc;path cc;
        save k,Val;numeric k,Val[];
        Val[1]:=Xmin;Val[2]:=Xmax;Val[3]:=100;
        k:=1;
        for i=t:
           Val[k]:=i;
           k:=k+1;
        endfor
        courbef_l(f,Val[1],Val[2],Val[3])
enddef;




vardef courbepoints(suffix f)(expr xmin, xmax, n) =  %Points non reliés
   save $,x;
   picture $;
   $=image(marquepoint((xmin,f(xmin)));
           for i=1 upto n-1:
              marquepoint((xmin+i*(xmax-xmin)/(n-1),f(xmin+i*(xmax-xmin)/(n-1))));
           endfor );
   $
enddef;

%vardef fonccourbe@#(expr x)=    %Ordonnée du point de la courbe @# d'abscisse x (dans %le repère utilisateur)
%   ypart pointcourbe@#(x)
%enddef;

vardef fonccourbe(expr p)(expr x)=    %Ordonnée du point de la courbe @# d'abscisse x (dans le repère utilisateur)
   ypart pointcourbe(p)(x)
enddef;

%vardef pointcourbe@#(expr x)=   %Point de la courbe @# d'abscisse x dans le repère %utilisateur
%save _P_,t;
%  numeric t; pair _P_;
%  (t,whatever)=@# intersectiontimes ((x,2*Ymin-Ymax)--(x,2*Ymax-Ymin));
%  if t=-1: _P_:=(x,0)
%  else: _P_:=point t of @#
%  fi;
%  _P_
%enddef;

vardef pointcourbe(expr p)(expr x)=   %Point de la courbe @# d'abscisse x dans le repère utilisateur
save _P_,t;
  numeric t; pair _P_;
  (t,whatever)=p intersectiontimes ((x,2*Ymin-Ymax)--(x,2*Ymax-Ymin));
  if t=-1: _P_:=(x,0)
  else: _P_:=point t of p
  fi;
  _P_
enddef;

vardef der(expr p)(expr x)=   %Fonction dérivée
   save pp,t,v,w,d;
   path pp;
   pair v,w;
   pp:=p;
   (t,whatever) = pp intersectiontimes ((x,2*Ymin-Ymax)--(x,2*Ymax-Ymin));
   v:=direction t of pp;
%   w:=unitvector(v transformed inverse _T);
   if xpart v = 0: d:=0 else: d:= (ypart v)/(xpart v) fi;
   d
enddef;

vardef deri@#(expr x)=   %Fonction dérivée
   save pp,t,v,w,d;
   path pp;
   pair v,w;
   pp:=@#;
   (t,whatever) = pp intersectiontimes ((x,2*Ymin-Ymax)--(x,2*Ymax-Ymin));
   v:=direction t of pp;
%   w:=unitvector(v transformed inverse _T);
   if xpart v = 0: d:=0 else: d:= (ypart v)/(xpart v) fi;
   d
enddef;

vardef intercourbes(suffix P)(expr p,q)= %renvoie les points d'intersections de p et q dans P[]
  save i,pp;
  numeric i;
  path pp;
  if unknown P: pair P[] fi;
  pp:=p;
  i:=1;
  forever:
   exitif xpart(pp intersectiontimes q)=-1;
   P[i]:=point xpart(pp intersectiontimes q) of pp;
   pp:=subpath(xpart(pp intersectiontimes q)+0.01,length pp) of pp;
                   %0.01(?!) pour que le point trouvé ne soit pas dans le sous-chemin...
   i:=i+1;
  endfor;
enddef;

vardef intercourbestimes(suffix T)(expr p,q)= %renvoie le "time" de q des points d'intersections de p et q dans T[]
  save i,pp;
  numeric i;
  path pp;
  pp:=p;
  i:=1;
  forever:
   exitif xpart(pp intersectiontimes q)=-1;
   T[i]:=ypart(pp intersectiontimes q);
   pp:=subpath(xpart(pp intersectiontimes q)+0.01,length pp) of pp;
                   %0.01(?!) pour que le point trouvé ne soit pas dans le sous-chemin...
   i:=i+1;
   exitif i=10;
  endfor;
enddef;

vardef ptantecedents(suffix P)(expr y,p)=
  intercourbes(P,(Xmin,y)--(Xmax,y),p)
enddef;

vardef antecedents(suffix P)(expr y,p)=
  pair PP[];
  numeric P[];
  ptantecedents(PP,y,p);
  for i=1 upto 10: P[i]:=xpart (PP[i]); endfor
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   INTERPOLATION   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Novembre 2017
vardef polynomelagrangept(text t)(expr x)=
  0
  for i=t:
    + ypart i
    for j=t:
      if i<>j: *((x-xpart j)/(xpart i - xpart j)) fi
    endfor
  endfor
enddef;

%Novembre 2017
vardef polynomelagrangeval(text t)(expr x)=
save tmp,k,j,ab,or;
pair tmp[];numeric k,j,ab,or;
k:=0;j:=0;
for i=t:
  if j=0: ab:=i else: or:=i; k:=k+1; tmp[k]=(ab,or) fi;
  j:=1-j;
endfor
polynomelagrangept(tmp[1] for ii=2 upto k: ,tmp[ii] endfor)(x)
enddef;

%Novembre 2017
vardef polynomelagrange(text t)(expr x)=
  save tmp;
  numeric tmp;
  for i=t:
    if pair i: tmp:=polynomelagrangept(t)(x)
    else: tmp:=polynomelagrangeval(t)(x)
    fi;
    exitif true;
  endfor
  tmp
enddef;


%Novembre 2017
vardef lagrange(text t)(text q)=
  vardef _tmp_poly_lag(expr x)= polynomelagrange(t)(x) enddef;
  courbefonc(_tmp_poly_lag)(q)
enddef;


%Novembre 2017
vardef polynomehermitetriplet(text t)(expr x)=
%t=(x1,y1,z1),(x2,y2,z2)...
%H(X)=Σqi(X)Pi(X) avec qi(X)=Li(X)² et Pi(X)=yi+(X-xi)(zi-qi'(xi)yi)
%qi'(xi)=Σ2/(xi-xj)
0
for i=t:
   +
   (greenpart i+(x-redpart i)*(bluepart i - 
   (0
   for j=t:   %qi'(xi)
     if i<>j: + 2/(redpart i - redpart j) fi
   endfor
   )
   *greenpart i))
   for j=t:  % qi(X)
     if i<>j: *(((x-redpart j)/(redpart i - redpart j))**2) fi
   endfor
endfor
enddef;

%Novembre 2017
vardef polynomehermiteptder(text t)(expr x)=
save tmp,k,j,re,gr,bl,i;
color tmp[];numeric k,j,re,gr,bl,i;
k:=0;j:=0;
for i=t:
   if pair i: re:= xpart i; gr:= ypart i else: bl:=i; k:=k+1; tmp[k]=(re,gr,bl) fi;
endfor
polynomehermitetriplet(tmp[1] for ii=2 upto k: ,tmp[ii] endfor)(x)
enddef;

%Novembre 2017
vardef polynomehermite(text t)(expr x)=
  save tmp;
  numeric tmp;
  for i=t:
    if color i: tmp:=polynomehermitetriplet(t)(x)
    else: tmp:=polynomehermiteptder(t)(x)
    fi;
    exitif true;
  endfor
  tmp
enddef;


%Novembre 2017
vardef hermite(text t)(text q)=
  vardef _tmp_poly_her(expr x)= polynomehermite(t)(x) enddef;
  courbefoncl(_tmp_poly_her)(q)
enddef;


% Novembre 2017
vardef spcuhezz (expr t)= 2*(t**3)-3*(t**2)+1 enddef;
vardef spcuheuz (expr t)= t**3-2*(t**2)+t enddef;
vardef spcuhezu (expr t)= -2*(t**3)+3*(t**2) enddef;
vardef spcuheuu (expr t)= t**3-t**2 enddef;


% Novembre 2017
vardef splinecubique (expr xa,ya,da,xb,yb,db)(expr x)=
% On donne deux points et les dérivées en ces points
  save t;
  numeric t;
  t=(x-xa)/(xb-xa);
  (spcuhezz(t))*ya + (spcuheuz(t))*da*(xb - xa) + (spcuhezu(t))*yb + (spcuheuu(t))*db*(xb - xa)
enddef;

% Novembre 2017
vardef splineinterpolderfonct_coul(text t)(expr x)=
% On donne points et dérivées sous forme de triplets (x,y,y')
  save pta,ptb,prem,sortie;
  color pta,ptb;
  boolean prem,sortie;
  prem:=true;sortie:=false;
  for i=t:
    if prem: ptb:=i; prem:=false
    else:
       pta:=ptb;
       ptb:=i;
       if x<= redpart ptb: sortie:=true fi;
    fi;
    exitif sortie;
  endfor
  splinecubique(redpart pta, greenpart pta, bluepart pta,redpart ptb, greenpart ptb, bluepart ptb)(x)
enddef;

% Novembre 2017
vardef splineinterpolderfonct_pt(text t)(expr x)=
% On donne points et dérivées sous forme de liste (A,y'A,B,y'B...)
   save tmp,k,re,gr,bl;
   color tmp[];
   numeric re,gr,bl,k;
   k:=0;
   for i=t:
     if pair i: k:=k+1; re:= xpart i; gr:= ypart i
     else: bl:=i; tmp[k]:=(re,gr,bl)
     fi;
   endfor
   splineinterpolderfonct_coul(tmp[1] for ii=2 upto k: ,tmp[ii] endfor)(x)
enddef;

% Novembre 2017
vardef splineinterpolderfonct_listeval(text t)(expr x)=
% On donne points et dérivées sous forme de liste de valeurs (xA,yA,y'A,xB,yB,y'B...)
   save tmp,k,j,re,gr,bl;
   color tmp[];
   k:=0;j:=0;
   for i=t:
      if j=0: re:= i; j:=1
      elseif j=1: gr:=i; j:=2
      else: bl:=i; tmp[k]:=(re,gr,bl); j:=0;k:=k+1
      fi;
    endfor
   splineinterpolderfonct_coul(tmp[0] for ii=1 upto k-1: ,tmp[ii] endfor)(x)
enddef;


% Novembre 2017
vardef splinederfonct(text t)(expr x)=
  save tmp;
  numeric tmp;
  for i=t:
    if color i: tmp:=splineinterpolderfonct_coul(t)(x)
    elseif pair i: tmp:=splineinterpolderfonct_pt(t)(x)
    else: tmp:=splineinterpolderfonct_listeval(t)(x)
    fi;
    exitif true;
  endfor
  tmp
enddef;

% Novembre 2017
vardef splineder(text t)(text q)=
  vardef _tmp_poly_splineder(expr x)= splinederfonct(t)(x) enddef;
  courbefoncl(_tmp_poly_splineder)(q)
enddef;



%Mars 2019
vardef deriveesplinecubique(suffix k)(suffix xs,ys)(expr n)=
% renvoie les dérivées pour interpolation avec splines cubiques
% source : https://en.wikipedia.org/wiki/Spline_interpolation
% xs[] et ys[] sont des listes, n est le nombre de valeurs
% Renvoie les dérivées dans la liste k[]
  if not derex[0]:
    (2k[0] + k[1])/(xs[1] - xs[0]) = 3(ys[1]-ys[0])/((xs[1]-xs[0])**2)
  fi;
  for i=1 upto n-1:
     if not derex[i]:
        k[i-1]/(xs[i] - xs[i-1]) + 2k[i]/(xs[i] - xs[i-1]) + 2k[i]/(xs[i+1] - xs[i]) + k[i+1]/(xs[i+1] - xs[i]) = 3(ys[i] - ys[i-1])/((xs[i]-xs[i-1])**2) + 3(ys[i+1] - ys[i])/((xs[i+1]-xs[i])**2)
     fi;
  endfor
  if not derex[n]:
     (k[n-1] + 2k[n])/(xs[n]-xs[n-1]) = 3(ys[n]-ys[n-1])/((xs[n]-xs[n-1])**2)
  fi;
enddef;


% Mars 2019
vardef splineinterpolfonct_pt(text t)(expr x)=
% t est la liste A,[y'1,]B,[y'2,]C,[y'3,]... 
  save xa,ya,dd,n,derex;
  numeric xa[],ya[],dd[],n;
  boolean derex[];
  n:=-1;
  for i=t:
    if numeric i: 
       dd[n]:=i;derex[n]:=true;
    else:
       n := n + 1;
       derex[n]:=false;
       xa[n] := xpart i; ya[n] := ypart i
    fi;
  endfor
  deriveesplinecubique(dd)(xa,ya)(n);
  splinederfonct((xa[0],ya[0],dd[0]) for ii=1 upto n: , (xa[ii],ya[ii],dd[ii]) endfor)(x)
enddef;

vardef splineinterpolfonct_listeval(text t)(expr x)=
% t est la liste des valeurs (x1,y1,x2,y2,x3,y3...)
  save xa,ya,tmp,k,j;
  numeric xa,ya,k,j;
  pair tmp[];
  k:=0;j:=0;
  for i=t:
    if j=0: xa:= i
    else: ya:=i; tmp[k]:= (xa,ya); k:= k+1
    fi;
    j:=1-j;
  endfor
  splineinterpolfonct_pt(tmp[0] for ii=1 upto k-1: ,tmp[ii] endfor)(x)
enddef;

% Novembre 2017
vardef splinefonct(text t)(expr x)=
  save tmp;
  numeric tmp;
  for i=t:
    if pair i: tmp:=splineinterpolfonct_pt(t)(x)
    else: tmp:=splineinterpolfonct_listeval(t)(x)
    fi;
    exitif true;
  endfor
  tmp
enddef;

% Novembre 2017
vardef spline(text t)(text q)=
  vardef _tmp_poly_spline(expr x)= splinefonct(t)(x) enddef;
  courbefonc(_tmp_poly_spline)(q)
enddef;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   TANGENTES   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numeric longtan;
longtan:=20;

vardef tangentefleche@# (expr p)(text tt) =
   save t,$,pp,i,N,c;
pair senstan.gauche,senstan.droite,senstan.double;
senstan.gauche=(1,0);senstan.droite=(0,1);senstan.double=(1,1);
   numeric N[],i,c;
   i:=0;
   pair _v,M;
   picture $;
   for n=tt: i:=incr i ; N[i]:=n; endfor;
   if i=1: N2:=longtan fi;
   (t,whatever) = p intersectiontimes ((N1,Ymin)--(N1,Ymax));
   M:=point t of p;
   c:= ypart direction t of p;
   _v:=N2*unitvector(direction t of p);
   $=image(%
		marquepoint(M);
		if (xpart senstan@#) <> 0: olddrawarrow _cart(M)--_cart(M)-_v*(xpart senstan@#) fi;
		if (ypart senstan@#) <> 0: olddrawarrow _cart(M)--_cart(M)+_v*(ypart senstan@#) fi;
    );
%   $=image(%
%    drawdblarrow (M-)--(M+ )
%   );
   $
enddef;

vardef tangentedroite(expr p,a) =
 save $,m,M,t,c;
 path $;
 numeric m,t,c;
 pair M;
 (t,whatever) = p intersectiontimes ((a,Ymin)--(a,Ymax));
 M:=point t of p;
 c:= (ypart direction t of p)/(xpart direction t of p);
% $=M-_diag*unitvector(direction t of p)--M+_diag*unitvector(direction t of p);
 $=droite(c,ypart M - (xpart M)*c);
 $
enddef;

vardef tangente@#(suffix p)(text tt) =
  if str @#="": tangentedroite(p)(tt)
  else: tangentefleche@#(p)(tt)
  fi
enddef;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   CALCUL INTÉGRAL   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vardef entrecourbes(expr p,q,xmin,xmax)=
 buildcycle(droiteeqx(xmax),p,droiteeqx(xmin),q)
enddef;

vardef souscourbe(expr p,xmin,xmax)=
 buildcycle(droiteeqx(xmax),p,droiteeqx(xmin),(Xmin,0)--(Xmax,0))
enddef;


vardef rectangles@#(suffix p)(expr a,b,n)=   %Riemann, renvoie un chemin fermé
 save pas,_rect_;
 numeric pas;
 path _rect_;
 pas=(b-a)/n;
 _rect_ =
 for i=0 upto n-1:
   (a+i*pas,yO)--_segmrect_@#(p,a+i*pas,pas)--
 endfor
 (b,yO)--cycle;
 _rect_
enddef;

numeric _typerect.min,_typerect.max,_typerect.droite,_typerect.gauche;
_typerect.min=1;_typerect.max=2;_typerect.droite=3;_typerect.gauche=4;

vardef _segmrect_@#(suffix p)(expr a,pas)=
 save _sousp;
 path _sousp;
 _sousp=subpath (xpart(p intersectiontimes droiteeqx(a)),xpart(p intersectiontimes droiteeqx(a+pas))) of p;
 if _typerect@#=1: llcorner _sousp -- lrcorner _sousp
  elseif _typerect@#=2: ulcorner _sousp -- urcorner _sousp
   elseif _typerect@#=3: (pointcourbe(p)(a))--(pointcourbe(p)(a) shifted (pas,0))
    elseif _typerect@#=4: (pointcourbe(p)(a+pas) shifted (-pas,0))--(pointcourbe(p)(a+pas))
 fi
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   DROITES   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef droiteeqred(expr a,b)=
 save Pdr;
 pair Pdr[];
 if ((Ymin<a*Xmin+b) and (a*Xmin+b<Ymax)) or (a=0):
  Pdr[1]:=(Xmin,a*Xmin+b)
 elseif a>0:
  Pdr[1]:=((Ymin-b)/a,Ymin)
 else:
  Pdr[1]:=((Ymax-b)/a,Ymax)
 fi;
 if ((Ymin<a*Xmax+b) and (a*Xmax+b<Ymax)) or (a=0):
  Pdr[2]:=(Xmax,a*Xmax+b)
 elseif a>0:
  Pdr[2]:=((Ymax-b)/a,Ymax)
 else:
  Pdr[2]:=((Ymin-b)/a,Ymin)
 fi;
 Pdr[1]--
   if (Xmin<0) and (0<Xmax) and (Ymin<b) and (b<Ymax): (0,b)-- fi
 Pdr[2]
enddef;

vardef droiteeqx(expr c)=
  (c,Ymax)--
     if (Ymin<c) and (c<Ymax): (c,0)-- fi
  (c,Ymin)
enddef;


vardef droiteeqcart(expr a,b,c)=
 save $;
 path $;
 if b=0:
   $=droiteeqx(-c/a)
 else:
   $:=droiteeqred(-a/b,-c/b)
 fi;
 $
enddef;

vardef droiteeq(text t)=
  save $;
  path $;
  if long_texte(t)=1: $:= droiteeqx(t) fi;
  if long_texte(t)=2: $:= droiteeqred(t) fi;
  if long_texte(t)=3: $:= droiteeqcart(t) fi;
  $
enddef;


vardef droitept(expr A,B)=
  (_diag/abs(A-B))[B,A]--A--B--(_diag/abs(A-B))[A,B]
enddef;

vardef droite(text t)=
  save $;
  path $;
  for i=t:
    if pair i: $:=droitept(t) else: $:=droiteeq(t) fi;
    exitif true;
  endfor;
  $
enddef;


% demi-droites juin 2019
vardef demidroitept(expr A,B)=
  A--(_diag/abs(A-B))[A,B]
enddef;

vardef demidroiteangle(expr A,alpha)=
  demidroitept(A,A + dir alpha)
enddef;

vardef demidroite(text t)=
  save reptmp,repbool;
  path reptmp;
  boolean repbool;
  repbool:=false;
  for i=t:
    if numeric i: repbool := true fi;
  endfor
  if repbool:
    reptmp := demidroiteangle(t)
  else:
    reptmp := demidroitept(t)
  fi;
  reptmp
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   DEMI-PLANS   %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vardef demiplaninf(expr d)=
  save $,A,B;
  path $;
  pair A,B;
  if (point 0 of d) yscaled -1 < (point infinity of d) yscaled -1:
   A:=point 0 of d; B:= point infinity of d
   else: B:=point 0 of d; A:= point infinity of d
  fi;
  $=A--
  if ypart A>=ypart (Xmin,Ymax): (Xmin,Ymax)-- fi
  if ypart A>ypart (Xmin,Ymin): (Xmin,Ymin)-- fi
  if ypart B>ypart (Xmax,Ymin): (Xmax,Ymin)-- fi
  if ypart B>=ypart (Xmax,Ymax): (Xmax,Ymax)-- fi
  B--cycle;
  $
enddef;

vardef demiplansup(expr d)=
  save $,A,B;
  path $;
  pair A,B;
  if (point 0 of d) yscaled -1 < (point infinity of d) yscaled -1:
   A:=point 0 of d; B:= point infinity of d
   else: B:=point 0 of d; A:= point infinity of d
  fi;
  $=A--
  if ypart A<=ypart (Xmin,Ymin): (Xmin,Ymin)-- fi
  if ypart A<ypart (Xmin,Ymax): (Xmin,Ymax)-- fi
  if ypart B<ypart (Xmax,Ymax): (Xmax,Ymax)-- fi
  if ypart B<=ypart (Xmax,Ymin): (Xmax,Ymin)-- fi
  B--cycle;
  $
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   PROJECTIONS   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef projetex(expr A)=
  (xpart (A),yO)
enddef;

vardef projetey(expr A)=
  (xO,ypart (A))
enddef;

vardef Projectionx@#(text t)=
  save $,_a,_fg,dec;
  picture $,_fg;_fg:=nullpicture;
  numeric dec;dec=0;
  pair _a;
  $=image(%
   for i=t:
    if pair i: draw i--projetex(i);_a:=i elseif numeric i: dec:=i else: _fg:=thelabel.@#(i,projetex(_a)) fi;
   endfor;
   draw _fg shifted (0,dec));
  $
enddef;

def projectionx = Projectionx enddef;   %% Compatibilité

vardef Projectiony@#(text t)=
  save $,_a,_fg,dec;
  picture $,_fg;_fg:=nullpicture;
  numeric dec;dec=0;
  pair _a;
  $=image(%
   for i=t:
    if pair i: draw i--projetey(i);_a:=i elseif numeric i: dec:=i else: _fg:=thelabel.@#(i,projetey(_a)) fi;
   endfor;
   draw _fg shifted (dec,0));
  $
enddef;

def projectiony = Projectiony enddef;   %% Compatibilité

vardef Projectionaxes(text t)=
  save $,_A,n;
  picture $,_fg[];_fg1=_fg2=nullpicture;
  pair _A;
  numeric n,dd;n=1;dd=0;
   $=image(%
  for i=t: if numeric i: dd:=i fi; endfor
  for i=t:
   if not numeric i:
     if pair i: _A:=i;
     elseif n=1: if ypart _A > ypart _O_: draw projectionx.bot(_A,i,-dd) else: draw projectionx.top(_A,i,dd) fi;n:=n+1
     else: if xpart _A > xpart _O_: draw projectiony.lft(_A,i,-dd) else: draw projectiony.rt(_A,i,dd) fi;n:=n+1
     fi;
   fi;
  endfor;
  if n<=2: draw projectiony(_A) fi; if n=1: draw projectionx(_A) fi);
 $
enddef;

def projectionaxes = Projectionaxes enddef;   %% Compatibilité

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%   SUITES   %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef suite(suffix u)(expr deb,fin)=
  save $;
  picture $;
  $=courbepoints(u,deb,fin,fin-deb+1);
  $
enddef;

vardef _suiterecfonc_(expr x)=0 enddef;
numeric _suiterecdeb_, _suiterecfin_, _suiterecinit_;

vardef suiterec(suffix f)(expr deb,fin,init)= %escalier un+1=f(un), u_deb=init
  vardef _suiterecfonc_(expr x)=f(x) enddef;
  _suiterecdeb_:=deb;_suiterecfin_:=fin;_suiterecinit_:=init;	%stocke les valeurs pour réutilisation
  save $,un;
  path $;
  numeric un;
  un=init;
  $=(un,yO)
  for i=deb upto fin-1:
    --(un,f(un))--(f(un),f(un))
    hide(un:=f(un))
  endfor;
  $
enddef;

vardef suiterecprojx@#(text t)=   	%projections et étiquettes sur axe x, utilise les valeurs de suiterec
  save $,deb,fin,nom,_qw_,un,lab;
  picture $;
  string nom,lab;
  numeric deb,fin,_qw_,un,_prec;
  deb:=_suiterecdeb_;fin:=_suiterecfin_;un:=_suiterecinit_;
  _qw_:=1;
  for i=t:			%récupère les éventuelles nouvelles valeurs pour deb et fin
    if _qw_ =1: if string i: nom:=i else: nom:="_val";_prec:=i fi;
    elseif _qw_=2:
    		deb:=i;
    		for j=_suiterecdeb_ upto i-1:
    		  un:=_suiterecfonc_(un);
    		endfor
    elseif _qw_=3: fin:=i
    fi;
    _qw_:=_qw_+1;
  endfor;
  $=image(%
  if nom="":
    if deb=_suiterecdeb_:
      deb:=deb+1;un:=_suiterecfonc_(un)
    fi;
    for i=deb upto fin:
      draw projectionx.@#((un,un));
      un:=_suiterecfonc_(un);
    endfor
  elseif nom="_val":
    if deb=_suiterecdeb_:
      draw etiquettex.@#(arrondidec(_prec,un));
      deb:=deb+1;un:=_suiterecfonc_(un)
    fi;
    for i=deb upto fin:
      lab:="$\num{" &  decimal(arrondidec(_prec,un)) & "}$";
      draw projectionx.@#((un,un),lab,-taillegrad);
      un:=_suiterecfonc_(un);
    endfor
  else:
    if deb=_suiterecdeb_:
      label.@#("$" & nom & "_{" & decimal(deb) & "}" & "$",(un,yO));
      deb:=deb+1;un:=_suiterecfonc_(un)
    fi;
    for i=deb upto fin:
      lab:="$" & nom & "_{" & decimal(i) & "}" & "$";
      draw projectionx.@#((un,un),lab);
      un:=_suiterecfonc_(un);
    endfor
  fi);
  $
enddef;

vardef suiterecprojy@#(text t)=   	%projections et étiquettes sur axe y, utilise les valeurs de suiterec
  save $,deb,fin,nom,_qw_,un,lab;
  picture $;
  string nom,lab;
  numeric deb,fin,_qw_,un,_prec;
  deb:=_suiterecdeb_;fin:=_suiterecfin_;un:=_suiterecinit_;
  _qw_:=1;
  for i=t:				%récupère les éventuelles nouvelles valeurs pour deb et fin
    if _qw_ =1: if string i: nom:=i else: nom:="_val";_prec:=i fi;
    elseif _qw_=2:
    		deb:=i;
    		for j=_suiterecdeb_ upto i-2:
    		  un:=_suiterecfonc_(un);
    		endfor
    elseif _qw_=3: fin:=i
    fi;
    _qw_:=_qw_+1;
  endfor;
  $=image(%
  if nom="":
    if deb=_suiterecdeb_:
      deb:=deb+1;
    fi;
    for i=deb-1 upto fin-1:
      draw projectiony.@#((un,_suiterecfonc_(un)));
      un:=_suiterecfonc_(un);
    endfor
  elseif nom="_val":
    if deb=_suiterecdeb_:
      deb:=deb+1;
    fi;
    for i=deb-1 upto fin-1:
      lab:="$\num{" &  decimal(arrondidec(_prec,un)) & "}$";
      draw projectiony.@#((un,_suiterecfonc_(un)),lab,-taillegrad);
      un:=_suiterecfonc_(un);
    endfor
  else:
    if deb=_suiterecdeb_:
      deb:=deb+1;
    fi;
    for i=deb-1 upto fin-1:
      lab:="$" & nom & "_{" & decimal(i+1) & "}" & "$";
      draw projectiony.@#((un,_suiterecfonc_(un)),lab);
      un:=_suiterecfonc_(un);
    endfor
  fi);
  $
enddef;


vardef suiterecproj(text t)=
 save $;
 picture $;
 $=image(%
 draw suiterecprojx.bot(t);
 draw suiterecprojy.lft(t));
 $
enddef;


% Octobre 2022 - SuiteRec figure complète
string suite_nom,suite_labx,suite_laby,suite_posx,suite_posy,suite_styleproj;
boolean suite_affx,suite_affy;
numeric suite_arrondi;
suite_nom:="u";
suite_arrondi:=1;
suite_labx:="nom";
suite_laby:="";
suite_affx:=true;
suite_affy:=true;
suite_posx:="bot";
suite_posy:="lft";
suite_styleproj:="dashed evenly";

vardef Suiterec(suffix f)(expr deb,fin,init) text t=
  save tmpfig,tmpsr;
  path tmpsr;
  picture tmpfig;
  tmpfig:=image(%
    tmpsr:= suiterec(f,deb,fin,init);
    if suite_labx="val":
       numeric tmpsreclab;
       tmpsreclab:=suite_arrondi
    else:
       string tmpsreclab;
       if suite_labx="nom":
          tmpsreclab:=suite_nom
       else:
          tmpsreclab:=""
       fi
    fi;
    if suite_affx: 
       draw scantokens("suiterecprojx."& suite_posx)(tmpsreclab) t scantokens(suite_styleproj)
    fi;
    if suite_laby="val":
       numeric tmpsreclab;
       tmpsreclab:=suite_arrondi
    else:
       string tmpsreclab;
       if suite_laby="nom":
          tmpsreclab:=suite_nom
       else:
          tmpsreclab:=""
       fi
    fi;
    if suite_affy: draw scantokens("suiterecprojy."& suite_posy)(tmpsreclab) t scantokens(suite_styleproj) fi;
    draw tmpsr t;
  );
  tmpfig
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   INTERVALLES   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef bornex[](expr val)=
 ((1mm,-2mm)--(0,-2mm)--(0,2mm)--(1mm,2mm)) if @<0: xscaled -1 fi shifted _cart(val,xO)
enddef;

vardef borney[](expr val)=
 ((2mm,1mm)--(2mm,0)--(-2mm,0)--(-2mm,1mm)) if @<0: yscaled -1 fi shifted _cart(yO,val)
enddef;

pair sensborne.FF,sensborne.FO,sensborne.OF,sensborne.OO;
sensborne.FF=-sensborne.OO=(1,-1);sensborne.FO=-sensborne.OF=(1,1);

numeric int_ep;
int_ep:=2bp;

vardef intervallex@#(expr vmin,vmax)=
  save $;
  picture $;
  $=image(%
   olddraw _cart(vmin,xO)--_cart(vmax,xO) withpen pencircle scaled int_ep;
   olddraw bornex[xpart sensborne@#](vmin) withpen pencircle scaled int_ep;
   olddraw bornex[ypart sensborne@#](vmax) withpen pencircle scaled int_ep);
  $
enddef;

vardef intervalley@#(expr vmin,vmax)=
  save $;
  picture $;
  $=image(%
   olddraw _cart(yO,vmin)--_cart(yO,vmax) withpen pencircle scaled int_ep;
   olddraw borney[xpart sensborne@#](vmin) withpen pencircle scaled int_ep;
   olddraw borney[ypart sensborne@#](vmax) withpen pencircle scaled int_ep);
  $
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%   STATISTIQUES   %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
%Obsolète 2022
%
% Deb
%

numeric _decboitemoustache_,_largboitemoustache_;


vardef boitemoustache(text t)=
  _decboitemoustache_:=1.5cm;
  _largboitemoustache_:=1cm;
  save $,n;
   picture $;
   numeric n,_valeursboitemoustache_[];
   n:=1;
   for i=t:
     _valeursboitemoustache_[n]:=i;
     n:=n+1;
   endfor;
   if known _valeursboitemoustache_[6]: _decboitemoustache_:=_valeursboitemoustache_[6] fi;
   if known _valeursboitemoustache_[7]: _largboitemoustache_:=_valeursboitemoustache_[7] fi;
   $=image(%
   oldfill fullcircle scaled 3 shifted _cart(_valeursboitemoustache_[1],yO);
   olddraw _cart(_valeursboitemoustache_[1],yO)--_cart(_valeursboitemoustache_[2],yO);
   olddraw _cart(_valeursboitemoustache_[2],yO) shifted (0,-_largboitemoustache_/2)--_cart(_valeursboitemoustache_[2],yO) shifted (0,_largboitemoustache_/2)--_cart(_valeursboitemoustache_[4],yO) shifted (0,_largboitemoustache_/2)--_cart(_valeursboitemoustache_[4],yO) shifted (0,-_largboitemoustache_/2)--cycle;
   olddraw _cart(_valeursboitemoustache_[3],yO) shifted (0,-_largboitemoustache_/2)--_cart(_valeursboitemoustache_[3],yO) shifted (0,_largboitemoustache_/2);
   olddraw _cart(_valeursboitemoustache_[4],yO)--_cart(_valeursboitemoustache_[5],yO);
   oldfill fullcircle scaled 3 shifted _cart(_valeursboitemoustache_[5],yO)%
   ) shifted (0,_decboitemoustache_);
   $
enddef;

vardef projboitemoustache@#(text t)=
   save $,n,_lab_,_comp_,A,tmp;
   pair A;
   picture $;
   string _lab_[];
   numeric n_comp_,tmp;
   n:=long_texte(t);
   if n=0:
     _lab_[1]:="$X_{min}$";
     _lab_[2]:="$Q_1$";
     _lab_[3]:="$M_e$";
     _lab_[4]:="$Q_3$";
     _lab_[5]:="$X_{max}$";
   elseif n=1:
     if t=0: for i=1 upto 5: _lab_[i]:=""; endfor
     else: 
       tmp:=floor(t);
       for i=1 upto 5:
          _lab_[i]:="\num{" & decimal(arrondi(10**tmp,_valeursboitemoustache_[i])) & "}";
       endfor
     fi
   else: _comp_:=1; for i=t:
                       if numeric i: _lab_[_comp_]:="\num{" & decimal(i) & "}"
                         elseif string i: _lab_[_comp_]:=i
                         else: _lab_[_comp_]:=i
                       fi;
                       _comp_:=_comp_+1;
                    endfor
   fi;
   $=image(%
   for i=1 upto 5:
     if (i=1) or (i=5):
       A:=_cart(_valeursboitemoustache_[i],yO) shifted(0,_decboitemoustache_)
       else: A:=_cart(_valeursboitemoustache_[i],yO) shifted(0,_decboitemoustache_-_largboitemoustache_/2)
     fi;
     draw projectionx@#(A transformed inverse _T,_lab_[i]);
   endfor);
   $
enddef;

%
% Fin
%
% Obsolète 2022
%

% Novembre 2022 - Nouvelle version diagramme en boite
numeric boite_dec,boite_larg;
string boite_lab,boite_lab_[],boite_pos,boite_styleproj;
boolean boite_points,boite_proj;
boite_dec:=1.5;
boite_larg:=1;
boite_points:=true;
boite_pos:="bot";
boite_styleproj:="dashed evenly";
boite_proj:=true;
boite_lab:="nom";


def def_boite_lab(text t)=
  save k;
  numeric k;
  k:=0;
  for i=t:
     k := k+1;
     if string i: boite_lab_[k]:=i
     elseif numeric i: boite_lab_[k]:="\num{" & decimal(i) & "}"
     fi;
  endfor
enddef;

def_boite_lab("$X_{\text{min}}$","$Q_1$","$M$","$Q_3$","$X_{\text{max}}$");

numeric _valeursboitemoustache_[];
def def_boite_val(text t)=
   save n;
   numeric n;
   n:=0;
   for i=t:
     n:=n+1;
     _valeursboitemoustache_[n]:=i;
   endfor;
enddef;

vardef diagrammeboite(text t)=
  save tmp;
   path tmp;
   def_boite_val(t);
   tmp := ((_valeursboitemoustache_[1],yO)--(_valeursboitemoustache_[2],yO)--
          (_valeursboitemoustache_[2],yO-0.5boite_larg)--(_valeursboitemoustache_[3],yO-0.5boite_larg)--
          (_valeursboitemoustache_[3],yO+0.5boite_larg)--(_valeursboitemoustache_[3],yO-0.5boite_larg)--
          (_valeursboitemoustache_[4],yO-0.5boite_larg)--(_valeursboitemoustache_[4],yO)--
          (_valeursboitemoustache_[5],yO)--(_valeursboitemoustache_[4],yO)--
          (_valeursboitemoustache_[4],yO+0.5boite_larg)--(_valeursboitemoustache_[2],yO+0.5boite_larg)--
          (_valeursboitemoustache_[2],yO)--cycle) shifted (0,boite_dec);
   tmp
enddef;



vardef Diagrammeboite(text s) text t=
  save figtmp;
  picture figtmp;
  def_boite_val(s);
  if boite_lab="":
     def_boite_lab("","","","","")
  elseif boite_lab="val":
     def_boite_lab(_valeursboitemoustache_[1],_valeursboitemoustache_[2],_valeursboitemoustache_[3],_valeursboitemoustache_[4],_valeursboitemoustache_[5])
  fi;
  figtmp:=image(%
    draw diagrammeboite() t;
    if boite_points: marquepoint((_valeursboitemoustache_[1],yO+boite_dec)) t;
                     marquepoint((_valeursboitemoustache_[5],yO+boite_dec)) t
    fi;
    if boite_proj:
      for i=1,5:
        draw scantokens("Projectionx." & boite_pos)((_valeursboitemoustache_[i],yO+boite_dec),boite_lab_[i]) t scantokens(boite_styleproj);
      endfor
      for i=2,3,4:
        draw scantokens("Projectionx." & boite_pos)((_valeursboitemoustache_[i],yO+boite_dec-0.5boite_larg),boite_lab_[i]) t scantokens(boite_styleproj);
      endfor
    fi
    );
  figtmp
enddef;


numeric batons_diampoints;
batons_diampoints:=5;

vardef diagrammebatons(text t) =
   save $;
   picture $;
   numeric tmp_taillepoint;
   string tmp_marque_p;
   tmp_taillepoint:=taillepoint;
   tmp_marque_p := marque_p;
   $=image(%
     interim linecap:=butt;
     marque_p:="plein";
     taillepoint:=batons_diampoints*0.5;
     for i=t:
        draw i--projetex(i);
        marquepoint(i);
     endfor
     );
   taillepoint:=tmp_taillepoint;
   marque_p := tmp_marque_p;
   $
enddef;


numeric barres_larg;
barres_larg := 20;

vardef barrei(expr p)=
   (projetex(_cart(p)-0.5(barres_larg,0))--(_cart(p)-0.5(barres_larg,0))--(_cart(p)+0.5(barres_larg,0))--projetex(_cart(p)+0.5(barres_larg,0))) transformed inverse _T
enddef;

vardef diagrammebarres(text t)=
   save tmp;
   path tmp;
   tmp:= for i=t: barrei(i)-- endfor cycle;
   tmp
enddef;

%octobre 2016
%diagramme en batons de la loi binomiale
vardef diagrammebinomiale(expr n,p)=
   save tmp;
   picture tmp;
   tmp:=scantokens("diagrammebatons("&
           "(0," & decimal(binomiale(n,p,0)) & ")" &
           for i=1 upto n: ",(" & decimal(i) & "," & decimal(binomiale(n,p,i)) & ")" &
           endfor
           ")");
   tmp
enddef;

%mai 2017
%diagramme en batons de la loi uniforme discrète
vardef diagrammeuniforme(expr n,m)=
   save tmp;
   picture tmp;
   tmp:=scantokens("diagrammebatons("&
           "(" & decimal(n) & "," & decimal(1/(m-n+1)) & ")" &
           for i=n+1 upto m: ",(" & decimal(i) & "," & decimal(1/(m-n+1)) & ")" &
           endfor
           ")");
   tmp
enddef;

%mai 2017
%diagramme en batons de la loi géométrique
vardef diagrammegeometrique(expr p)=
   save tmp;
   picture tmp;
   tmp:=scantokens("diagrammebatons("&
           "(1," & decimal(p) & ")" &
           for i=2 upto Xmax: ",(" & decimal(i) & "," & decimal(((1-p)**(i-1))*p) & ")" &
           endfor
           ")");
   tmp
enddef;

%mai 2017
%diagramme en batons de la loi de Poisson
vardef diagrammepoisson(expr lambda)=
   save tmp;
   picture tmp;
   tmp:=scantokens("diagrammebatons("&
           "(0," & decimal(poisson(lambda,0)) & ")" &
           for i=1 upto Xmax: ",(" & decimal(i) & "," & decimal(poisson(lambda,i)) & ")" &
           endfor
           ")");
   tmp
enddef;


% mai 2017 - modifiée novembre 2022
% courbe de la densité de la loi normale
vardef densitenormale(expr mu,sigma)(text t)=
  vardef tmpdens(expr x)=
    (1/(sigma*sqrt(2*pi)))*exp(-0.5*(((x-mu)/sigma)**2))
  enddef;
  courbefonc(tmpdens)(t)
enddef;

% Mai 2017
% courbe de la densité de la loi exponentielle
vardef densiteexponentielle(expr lambda)(text t)=
  vardef tmpdens(expr x)=
    lambda*exp(-lambda*x)
  enddef;
   save k,Val;numeric k,Val[];
   Val[1]:=0;Val[2]:=xmaxf;Val[3]:=nbf;
   k:=1;
   for i=t:
      Val[k]:=i;
      k:=k+1;
   endfor
   if Val[1]<0: Val[1]:=0 fi;
   courbef_(tmpdens,Val[1],Val[2],Val[3])
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%   POLYGONES   %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vardef polygone(text t)=
  save poly;
  path poly;
  poly := 
    for i=t:
      i--
    endfor
    cycle;
  poly
enddef;

vardef triangle(expr A,B,C)=
    polygone(A,B,C)
enddef;

vardef parallelogramme(expr A,O,B)=
  O--A--(A+B-O)--B--cycle
enddef;

vardef polygoneregulier(expr A,B,n)=
  save poly,centre;
  path poly;pair centre;
  _cart(B)-centre = (_cart(A)-centre) rotated (360/n);
  poly:= _cart(A)
     for i = 1 upto n-1:
       -- (_cart(A) rotatedaround(centre,(i*360)/n))
     endfor
     --cycle;
  poly transformed inverse _T
enddef;

vardef sommetpolygoneregulier(expr A,B,n,i)=
  save $,centre;
  pair $,centre;
  _cart(B)-centre = (_cart(A)-centre) rotated (360/n);
  $= _cart(A) rotatedaround(centre,((i-1)*360)/n);
  $ transformed inverse _T
enddef;

vardef equilateral(expr A,B)=
  polygoneregulier(A,B,3)
enddef;

vardef carre(expr A,B)=
  polygoneregulier(A,B,4)
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%   CERCLES ET ARCS   %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef arccercle(expr A,O,B)=
  path tmp,arc;
  tmp:=fullcircle scaled (2*(abs(_cart(A)-_cart(O)))) rotated angle(_cart(A)-_cart(O)) shifted _cart(O);
  if angle(_cart(B)-_cart(O))>angle(_cart(A)-_cart(O)):
    arc:=subpath(0,(angle(_cart(B)-_cart(O))-angle(_cart(A)-_cart(O)))*(length tmp)/360) of tmp;
  else:
    arc:=subpath(0,(360-(angle(_cart(A)-_cart(O))-angle(_cart(B)-_cart(O))))*(length tmp)/360) of tmp;
  fi
  arc transformed inverse _T
enddef;

vardef cerclecr(expr O,r)=
  fullcircle scaled (2*r*Ux) shifted _cart(O) transformed inverse _T
enddef;

vardef cerclecp(expr O,A)=
%  fullcircle scaled (2*abs(_cart(A)-_cart(O))) shifted _cart(O) transformed inverse _T
  fullcircle scaled (2*abs(_cart(A)-_cart(O))) rotated angle(_cart(A)-_cart(O)) shifted _cart(O) transformed inverse _T
enddef;

vardef cerclecir(expr A,B,C)=
  save O;pair O;
  _cart(O)-0.5[_cart(A),_cart(B)]=whatever * (_cart(B)-_cart(A)) rotated 90;
  _cart(O)-0.5[_cart(A),_cart(C)]=whatever * (_cart(C)-_cart(A)) rotated 90;
  cerclecp(O,A)
enddef;

vardef cercle(text t)=
   save $;path $;
   save r;boolean r;
   r:=false;
   if long_texte(t)=3:
     $:=cerclecir(t)
   else:
     for i=t:
       if numeric i: r:=true fi;
     endfor
     if r: $:=cerclecr(t) else: $:=cerclecp(t) fi;
   fi;
   $   
enddef;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%   MARQUES ANGLES, SEGMENTS...   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


taille_marque_a:=0.4cm;
sep_marque_a:=1.5;


vardef arc_marqueangle(expr A,O,B,t)=
  arccercle((_cart(O)+t*unitvector(_cart(A)-_cart(O))) transformed inverse _T,O,(_cart(O)+t*unitvector(_cart(B)-_cart(O))) transformed inverse _T)
enddef;

vardef marqueanglegeo(expr A,O,B,n)=
  O--arc_marqueangle(A,O,B,taille_marque_a-(n-1)*sep_marque_a/2)
  for i=1 upto n-1:
   --reverse arc_marqueangle(A,O,B,taille_marque_a-(n-1)*sep_marque_a/2+(i-1)*sep_marque_a)--
   arc_marqueangle(A,O,B,taille_marque_a-(n-1)*sep_marque_a/2+i*sep_marque_a)
  endfor
  --cycle
enddef;


vardef marqueangleor(expr A,O,B)=
  arc_marqueangle(A,O,B,taille_marque_a)
enddef;

vardef marqueangle(text t)=
  if long_texte(t)=4: marqueanglegeo(t)
  else: marqueangleor(t)
  fi
enddef;

vardef nommeangle@#(expr A,O,B,p,n)=
  save P,a;pair P;path a;
  save marque_p; string marque_p;
  marque_p:="";
  if n>0: draw marqueangle(A,O,B,n) fi;
  a:=arc_marqueangle(A,O,B,taille_marque_a);
  P:= point (arctime 0.5*arclength a of a) of a;
  if str @# = "":
    nommepoint[angle (P-O)](P,p)
  else:
    nommepoint@#(P,p)
  fi
enddef;


echelle_marque_ad:=1;
taille_marque_ad:=0.3cm;

vardef marqueangledroit(expr A,O,B)=
  parallelogramme((_cart(O)+taille_marque_ad*echelle_marque_ad*unitvector(_cart(A)-_cart(O))) transformed inverse _T,O,(_cart(O)+taille_marque_ad*echelle_marque_ad*unitvector(_cart(B)-_cart(O))) transformed inverse _T)
enddef;

echelle_marque_s:=1;
taille_marque_s:=0.3cm;
angle_marque_s:=60;
sep_marque_s:=2;
string forme_marque_s;
forme_marque_s:="/";

vardef milieu(expr A,B)=
  0.5[A,B]
enddef;

vardef marqueuniquesegment_s(expr A,B)=
   ((-taille_marque_s*echelle_marque_s*unitvector(_cart(A)-_cart(B))/2)--(taille_marque_s*echelle_marque_s*unitvector(_cart(A)-_cart(B))/2)) rotated angle_marque_s
enddef;

vardef marqueuniquecroix_s(expr A,B)=
   ((-taille_marque_s*echelle_marque_s*unitvector(_cart(A)-_cart(B))/2) rotated 45 -- (taille_marque_s*echelle_marque_s*unitvector(_cart(A)-_cart(B))/2) rotated 45 -- (0,0) -- (-taille_marque_s*echelle_marque_s*unitvector(_cart(A)-_cart(B))/2) rotated -45 -- (taille_marque_s*echelle_marque_s*unitvector(_cart(A)-_cart(B))/2) rotated -45) 
enddef;

vardef marqueuniquevague_s(expr A,B)=
   ((-taille_marque_s*echelle_marque_s/2,0){1,-1}..(0,0){1,1}..{1,-1}(taille_marque_s*echelle_marque_s/2,0)) rotated (angle(_cart(A)-_cart(B))+angle_marque_s)
enddef;

vardef marqueuniquerond_s(expr A,B)=
   fullcircle scaled (0.5taille_marque_s*echelle_marque_s)
enddef;

%%
%% Figure de forme f, orientée en fonction de A--B
%%
vardef marqueunique_s(expr A,B,f)=
   if f="o":
      marqueuniquerond_s(A,B)
   elseif f="s":
      marqueuniquevague_s(A,B)
   elseif f="x":
      marqueuniquecroix_s(A,B)
   else:
      marqueuniquesegment_s(A,B)
   fi
enddef;


vardef marquepathpos(expr p,n,f,t)=
 save $,AA,BB;
 pair AA,BB;
 AA:= point 0 of p;
 BB:= point (length p) of p;
 picture $;
 $=image(
    for i=1 upto n:
      draw marqueunique_s(AA,BB,f) shifted ((i-1-(n-1)/2)*sep_marque_s*unitvector(_cart(BB)-_cart(AA))) transformed inverse _T;
    endfor
    );
  $ shifted ((point (arctime t*(arclength p) of p) of p) transformed _T)
enddef;

%%
%% n figures de forme f orientée en fonction de A--B et placées autour du milieu
%%
vardef marqueseg(expr A,B,n,f)=
 save $;
 picture $;
 $=image(
    for i=1 upto n:
      draw marqueunique_s(A,B,f) shifted ((i-1-(n-1)/2)*sep_marque_s*unitvector(_cart(B)-_cart(A))) transformed inverse _T;
    endfor
    );
  $ shifted milieu(_cart(A),_cart(B))
enddef;

%%
%% Macro utilisateur t=A,B,n,f ou t=A,B,f ou t=A,B,n
%%
vardef marquesegment(text t)=
 save n,N,Pti,Ptii,tmpfig,tmpforme,boolpath,tmppath;
 numeric n,N;
 picture tmpfig;
 string tmpforme;
 pair Pti,Ptii;
 path tmppath;
 boolean boolpath;
 boolpath:=false;
 tmpforme:=substring (0,1) of forme_marque_s;
 N:=0;
 n:=1;
 for i=t:
   if numeric i: n:=i 
   elseif string i: tmpforme:=substring (0,1) of i;n:=length i
   elseif path i: boolpath:=true;tmppath:=i
   fi;
 endfor
 if boolpath:
   tmpfig:= marquepathpos(tmppath,n,tmpforme,0.5)
 else:
   tmpfig:=image(%
   for i=t:
    N:=N+1;
    if ((N mod 2) = 1) and (pair i):
       Pti := i;
    elseif pair i:
       Ptii := i;
       draw marqueseg(Pti,Ptii,n,tmpforme);
    fi;
   endfor
     )
 fi;
  tmpfig
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   FIGURES GÉOMÉTRIQUES COMPLÈTES   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

boolean AffPoints, AffCotes, AffAngles, AffVect, AffDonnees;
AffPoints:=true; % 
AffCotes:=false;
AffAngles:=false;
AffVect:=true;
AffDonnees:=false;

string UnitCotes;
UnitCotes:="";

numeric ArrondiCotes,ArrondiAngles,EchelleAngles;
boolean AutoAngleDroit;
ArrondiCotes:=2;
ArrondiAngles:=1;
EchelleAngles:=0.8;
AutoAngleDroit:=false;

numeric rep_int_arr;
if numbersystem="decimal": rep_int_arr:= 9 else: rep_int_arr:=4 fi;

numeric rep_type_triangle; %0 : lll ; 1 : lla ; 2 : lla

vardef AffichageCoteSansUnite(expr nb,arr)=
   "\num[round-pad = false,round-mode=places,round-precision=" & decimal(arr) & "]{" & decimal(nb) & "}"
enddef;

vardef AffichageCoteAvecUnite(expr nb,arr,unit)=
   "\SI[round-pad = false,round-mode=places,round-precision=" & decimal(arr) & "]{" & decimal(nb) & "}{" & unit & "}"
enddef;

%vardef AffichageAngle(expr nb,arr)=
%   "\ang[round-mode=places,round-pad = false,drop-zero-decimal,round-precision=" & decimal(arr) & "]{" & decimal(nb) & "}"
%enddef;

numeric _tmp_puiss_dix_;
vardef AffichageAngle(expr nb,arr)=
   _tmp_puiss_dix_:=round(10**arr);
   "\ang{" & decimal(arrondi(_tmp_puiss_dix_,nb)) & "}"
%   "\ang{" & decimal(arrondi(10**arr,nb)) & "}"
enddef;


%
%%%%%%%%%%%%%%%%   Segments   %%%%%%%%%%%%%%%
vardef Segment(suffix AA,BB)(text t)(text s)=
  save ll_seg,posAA,anghh,count_tmp,chAA,chBB,chcote,pA,pB;
  numeric ll_seg,anghh,count_tmp;
  string chAA,chBB,chcote;
  pair pA,pB;
  pA:=AA;pB:=BB;
  count_tmp:=0;
  if AffCotes:
      if _tr_rapp_<>1:
        chcote:=decimal(_tr_rapp_*abs(BB-AA))
      else:
        chcote:=decimal(abs(BB-AA))
      fi;
     if UnitCotes<>"":
       chcote:="\SI[round-pad = false,round-mode=places,round-precision=" & decimal(ArrondiCotes) & "]{" & chcote & "}{" & UnitCotes & "}"
     else:
       chcote:="\num[round-pad = false,round-mode=places,round-precision=" & decimal(ArrondiCotes) & "]{" & chcote & "}";
     fi
  else:
     chcote:=""
  fi;
  if AffPoints:
     chAA:="$" & _chaine(AA) & "$";
     chBB:="$" & _chaine(BB) & "$";
  else:
    chAA:="";chBB:="";
  fi;
  for i=t:
    count_tmp:=count_tmp+1;
    if count_tmp = 1: chAA:=i
    elseif count_tmp=2: chBB:=i
    fi;
  endfor
  % On modifie les noms si on est dans une transformation et on définit les points
  if  _ext_tr_<>"":
    chAA:=aj_extr_trsf(AA,chAA);
    chBB:=aj_extr_trsf(BB,chBB);
  fi
  % On modifie les points si on est dans une transformation
    if _tr_bool_:
      def _transform_fig_= enddef;
      pA:=_transform_pt_(AA);
      pB:=_transform_pt_(BB);
    fi
  for i=s:
    chcote:=i;
  endfor
  SegmentComplet(pA,pB)(chAA,chBB,chcote)
enddef;



vardef SegmentComplet(expr AA,BB)(expr chAA,chBB,cotei)=
%% Les extrémités étant données, trace le segment, le nom des points et la cote
%% La position du nom est calculée automatiquement
  save tmp,anghh,tmpana,tmpanb,chA,chB;
  picture tmp;
  string chA,chB;
  chA:=chAA;chB:=chBB;
  numeric anghh,tmpana,tmpanb;
  anghh:=angle(BB-AA);
  tmpana=anghh+180 ;
  tmpanb=anghh ;
  tmp:=image(%
    draw AA--BB;
    nomme[tmpana](AA,chA) withpen currentpen;
    nomme[tmpanb](BB,chB) withpen currentpen;
    if xpart AA <= xpart BB:
       cote(AA,BB,cotei)
    else:
       cote(BB,AA,cotei)
    fi;
  );
  tmp
enddef;

numeric ll_seg;

vardef defSegmentL(suffix AA,BB)(text s)=
%% Définit les points AA et BB en fonction de la longueur, le point de départ et l'angle
  save count_tmp,anghh,posAA;
  numeric count_tmp,anghh;
  pair posAA;
  posAA:=(0,0);anghh:=0;ll_seg:=0;
  count_tmp:=0;
    for i=s:
      count_tmp:=count_tmp+1;
      if count_tmp=1:
        ll_seg:=i
      elseif pair i:
        posAA:=i
      else:
        anghh:=i
      fi;
    endfor
  % Point A
  if unknown AA: pair AA;AA:=posAA fi;
  % Point B
  if unknown BB:
        pair BB;
        BB:=(ll_seg,0) rotated anghh shifted AA;
  else: %du coup, on change la valeur de l et angh
        ll_seg:=abs(BB-AA);anghh:=angle(BB-AA);
  fi;
enddef;

vardef SegmentL(suffix AA,BB)(text s)(text t)(text q)=
%% Macro utilisateur, definit les points A et B et trace la figure complète en fonctions des données
  defSegmentL(AA,BB)(s);
  Segment(AA,BB)(t)(q)
enddef;


vardef segmentL(suffix AA,BB)(text s)=
%% Macro utilisateur, definit les points A et B et renvoie A--B
  defSegmentL(AA,BB)(s);
  AA--BB
enddef;

%%%%%%%%%%%%%%%%   Vecteurs   %%%%%%%%%%%%%%%


vardef VecteurComplet(expr AA,uu)(expr chAA,chBB,cotei) text t=
%% Les extrémités étant données, trace le vecteur, le nom des points et le nom du vecteur
%% La position des noms est calculée automatiquement
  save tmp,anghh,tmpana,tmpanb,marque_p_tmp;
  string marque_p_tmp;
  picture tmp;
  numeric anghh,tmpana,tmpanb;
  anghh:=angle(uu);
  tmpana=anghh+180 ;
  tmpanb=anghh ;
  marque_p_tmp:=marque_p;
  tmp:=image(%
    drawarrow AA--(AA+uu) t;
    nomme[tmpana](AA,chAA) t withpen currentpen;
    marque_p:="";
    nomme[tmpanb](AA+uu,chBB) t;
    marque_p:=marque_p_tmp;
    cote(AA,AA+uu,cotei) t;
  );
  tmp
enddef;

%vardef Vecteur(suffix

pair AA_vect,uu_vect;

vardef defVecteurL(text t)(text s)=
%% t peut être (u), (u,A) ou (u,A,B)
%% Définit le vecteur u et, éventuellement, les points A et B en fonction de la longueur, le point de départ et l'angle
  save anghh,count_tmp,ufixe,ll,posAA,anghh;
  numeric anghh,count_tmp,ll;
  pair posAA;
  boolean ufixe;
  ufixe:=false;
  count_tmp:=0;
  posAA:=(0,0);anghh:=0;ll:=0;
  for i=s:
    count_tmp:=count_tmp+1;
    if count_tmp=1:
      ll:=i
    elseif pair i:
      posAA:=i
    else:
      anghh:=i
    fi;
  endfor
  count_tmp:=0;
  forsuffixes i=t:
    count_tmp:=count_tmp+1;
    if count_tmp=1:
       if unknown i:
         i:=(ll,0) rotated anghh;
         uu_vect:=i
       else:
         uu_vect:=i
%         ll:=abs(uu_vect);anghh:=angle(uu_vect)
       fi;
    elseif count_tmp=2:
       if unknown i: i:=posAA fi;
       AA_vect:=i;
    else:
       i:=AA_vect+uu_vect
    fi;
  endfor
enddef;


vardef VecteurL(text q)(text s)(text t) text r=
%% Macro utilisateur, definit le vecteur u et éventuellement les points A et B et trace la figure complète en fonctions des données
  save tmp,posAA,count_tmp,chAA,chBB,chvect,AA_vect,uu_vect,ch;
  numeric count_tmp;
  pair posAA,AA_vect,uu_vect;
  string chAA,chBB,chvect,ch;
  picture tmp;
  defVecteurL(q)(s);
  chAA:="";chBB:="";chvect:="";
  count_tmp:=0;
  forsuffixes i=q:
    count_tmp:=count_tmp+1;
    ch:=_chaine(i);
    if count_tmp=1:
       if AffVect: chvect:="$\vv{" & ijmath(ch) & "}$" fi
    elseif count_tmp=2:
       if AffPoints: chAA:="$" & ch & "$" fi
    else:
       if AffPoints: chBB:="$" & ch & "$" fi
    fi;
  endfor
  count_tmp:=0;
  for i=t:
    count_tmp:=count_tmp+1;
    if count_tmp = 1: chvect:=i
    elseif count_tmp=2: chAA:=i
    else: chBB:=i
    fi;
  endfor
  VecteurComplet(AA_vect,uu_vect)(chAA,chBB,chvect) r
enddef;


%%%%%%%%%%%%%%%%   Triangles   %%%%%%%%%%%%%%%

pair rep_tri_AA,rep_tri_BB,rep_tri_CC;
numeric rep_tri_la,rep_tri_lb,rep_tri_lc,
        rep_tri_angA,rep_tri_angB,rep_tri_angC;

boolean rep_ad_A,rep_ad_B,rep_ad_C;

def cotepolyplacee(suffix ptA,ptB)(expr chcote,sensdirect)=
   if xpart ptA<= xpart ptB:
       if sensdirect:
         cote(ptA,ptB,chcote)
       else:
         cote.top(ptA,ptB,chcote)
       fi
   else:
       if sensdirect:
         cote.top(ptB,ptA,chcote)
       else:
         cote(ptB,ptA,chcote)
       fi
   fi
enddef;

def nommeangleaigu(expr B,A,C,chang,sensdirect)=
  if sensdirect:
     nomme(B,A,C,chang,1)
  else:
     nomme(C,A,B,chang,1)
  fi
enddef;


vardef TriangleComplet(expr pAA,pBB,pCC)(expr chAA,chBB,chCC,coteAB,coteBC,coteCA,angAA,angBB,angCC) =
%% Trace le triangle, les noms des points, les cotes et les angles
  save tmpang,tmpfig,tmpscale,sensdir,chA,chB,chC,AA,BB,CC;
  pair AA,BB,CC;
  AA:=pAA;BB:=pBB;CC:=pCC;
  string chA,chB,chC;
  chA:=chAA;chB:=chBB;chC:=chCC;
  numeric tmpang,tmpscale;
  boolean sensdir;
  if angle(CC rotatedaround(AA,-angle(BB-AA)) - AA)>0:
    sensdir:=true
  else:
    sensdir:=false
  fi;
  tmpscale:=defaultscale;
  picture tmpfig;
  tmpfig:=image(%
     draw triangle (AA,BB,CC);
     if AffPoints:
        tmpang:=angle(0.5BB+0.5CC-AA)+180;
        nomme[tmpang](AA,chAA) withpen currentpen;
        tmpang:=angle(0.5AA+0.5CC-BB)+180;
        nomme[tmpang](BB,chBB) withpen currentpen;
        tmpang:=angle(0.5BB+0.5AA-CC)+180;
        nomme[tmpang](CC,chCC) withpen currentpen;
     fi
     if AffDonnees:
        cote(AA,BB,coteAB);
        if rep_type_triangle=0: cote(BB,CC,coteBC) fi;
        if rep_type_triangle<=1: cote(CC,AA,coteCA) fi;
%     elseif AffCotes:
     else:
        cotepolyplacee(AA,BB)(coteAB,sensdir);
        cotepolyplacee(BB,CC)(coteBC,sensdir);
        cotepolyplacee(CC,AA)(coteCA,sensdir);
     fi
     defaultscale:=EchelleAngles;
     if AffDonnees and (rep_type_triangle<3):
        if rep_type_triangle>=1:
          if rep_ad_A and AutoAngleDroit:
             draw marqueangledroit(BB,AA,CC) withpen currentpen
          else:
             nommeangleaigu(BB,AA,CC,angAA,sensdir);
          fi;
        fi;
        if rep_type_triangle=2:
          if rep_ad_B and AutoAngleDroit:
             draw marqueangledroit(CC,BB,AA) withpen currentpen
          else:
             nommeangleaigu(CC,BB,AA,angBB,sensdir);
          fi;
        fi;
     else:
        if (nang>=1):
          if rep_ad_A and AutoAngleDroit:
             draw marqueangledroit(BB,AA,CC) withpen currentpen
          else:
             nommeangleaigu(BB,AA,CC,angAA,sensdir);
          fi;
        fi
        if (nang>=2):
          if rep_ad_B and AutoAngleDroit:
             draw marqueangledroit(CC,BB,AA) withpen currentpen
          else:
             nommeangleaigu(CC,BB,AA,angBB,sensdir);
          fi;
        fi
        if (nang=3):
          if rep_ad_C and AutoAngleDroit:
             draw marqueangledroit(AA,CC,BB) withpen currentpen
          else:
             nommeangleaigu(AA,CC,BB,angCC,sensdir);
          fi;
        fi
     fi
     defaultscale:=tmpscale;
  );
  tmpfig
enddef;

vardef defTriangleLLA(text t)(text s)=
  save nbsommet,count_tmp,posAA,anghh,sommetdefiniA,sommetdefiniB,sommetdefiniC,ang_tmp;
  numeric nbsommet,count_tmp,anghh,ang_tmp;
  boolean sommetdefiniA,sommetdefiniB,sommetdefiniC;
  pair posAA;
  nbsommet:=0;
  sommetdefiniA=sommetdefiniB=sommetdefiniC=false;
  rep_ad_A:=false;rep_ad_B:=false;rep_ad_C:=false;
  % on compte le nombre de sommets et on prend en compte lesquels sont définis
  forsuffixes i=t:
    nbsommet:=nbsommet+1;
    if nbsommet=1:
       if known i: sommetdefiniA:=true; rep_tri_AA:=i else: pair i fi
    elseif nbsommet=2:
       if known i: sommetdefiniB:=true; rep_tri_BB:=i else: pair i fi
    else:
       if known i: sommetdefiniC:=true; rep_tri_CC:=i else: pair i fi
    fi;
  endfor
  count_tmp:=0;
  posAA:=(0,0);anghh:=0;
  % on définit les valeurs de départ
  for i=s:
    count_tmp:=count_tmp+1;
    if count_tmp=1: rep_tri_lc:=i
    elseif count_tmp=2: rep_tri_lb:=i
    elseif count_tmp=3: rep_tri_angA:=i
    elseif pair i: if (not sommetdefiniA): posAA:=i fi
    else: anghh:=i
    fi;
  endfor
  % Angles droits ?
  if rep_tri_angA=90:
     rep_ad_A:=true
  elseif arrondi(10**(rep_int_arr),cosd(rep_tri_angA)*rep_tri_lc) = arrondi(10**(rep_int_arr),rep_tri_lb):
     rep_ad_C:=true
  elseif arrondi(10**(rep_int_arr),cosd(rep_tri_angA)*rep_tri_lb) = arrondi(10**(rep_int_arr),rep_tri_lc):
     rep_ad_B:=true
  fi; 
  count_tmp:=0;
  % on calcule les coordonnées des sommets
  forsuffixes i=t:
    count_tmp:=count_tmp+1;
    if count_tmp=1: %On définit A s'il n'existe pas. À partir de B ou C éventuellement
       if unknown i: 
          if sommetdefiniC and sommetdefiniB:
             rep_tri_la:=abs(rep_tri_CC-rep_tri_BB);
             if abs(sind(rep_tri_angA))<rep_tri_la/rep_tri_lc:
                ang_tmp:=180-Arcsind(rep_tri_lc*abs(sind(rep_tri_angA))/rep_tri_la);
                i:=(rep_tri_lc,0) rotated ((angle(rep_tri_CC-rep_tri_BB) mod 360) + ang_tmp);
             else:
                i:=(-rep_tri_lc,0) rotated anghh shifted rep_tri_BB;
                rep_tri_angA:=(angle(rep_tri_CC-i) mod 360)-(angle(rep_tri_BB-i) mod 360);
             fi;
             rep_tri_lb:=abs(rep_tri_CC-i);
          elseif sommetdefiniB: i:=(-rep_tri_lc,0) rotated anghh shifted rep_tri_BB
%          if sommetdefiniB: i:=(-rep_tri_lc,0) rotated anghh shifted rep_tri_BB
          elseif sommetdefiniC: i=(-rep_tri_lb,0) rotated (anghh+rep_tri_angA) shifted rep_tri_CC
          else: i:=posAA
          fi;
          rep_tri_AA:=i
       else:
          if sommetdefiniB: rep_tri_lc:= abs(rep_tri_BB-rep_tri_AA) fi;
          if sommetdefiniC: rep_tri_lb:= abs(rep_tri_CC-rep_tri_AA) fi;
          if sommetdefiniC and sommetdefiniB:
             rep_tri_angA:=angle(rep_tri_CC-rep_tri_AA)-angle(rep_tri_BB-rep_tri_AA)
          fi;
       fi;
    elseif count_tmp=2: %On définit B à partir de C si C existe
       if unknown i:
          if sommetdefiniC: 
              i:= (rep_tri_lc,0) rotated (angle(rep_tri_CC-rep_tri_AA)-rep_tri_angA) shifted rep_tri_AA
          else:
              i:=(rep_tri_lc,0) rotated anghh shifted rep_tri_AA
          fi;
          rep_tri_BB:=i
       fi;
     else:           %On définit C s'il n'existe pas
        if unknown i:
           i:=(rep_tri_lb,0) rotated (angle(rep_tri_BB-rep_tri_AA)+rep_tri_angA) shifted rep_tri_AA;
        rep_tri_CC:=i
        fi;
     fi;
  endfor
  rep_tri_la:=abs(rep_tri_CC-rep_tri_BB);
  rep_tri_angB:=(angle(rep_tri_AA-rep_tri_BB) mod 360)-(angle(rep_tri_CC-rep_tri_BB) mod 360);
  rep_tri_angC:=(angle(rep_tri_BB-rep_tri_CC) mod 360)-(angle(rep_tri_AA-rep_tri_CC) mod 360);
enddef;


vardef defTriangleLAA(text t)(text s)=
  save nbsommet,count_tmp,posAA,anghh,sommetdefiniA,sommetdefiniB,sommetdefiniC,ang_tmp;
  numeric nbsommet,count_tmp,anghh,ang_tmp;
  boolean sommetdefiniA,sommetdefiniB,sommetdefiniC;
  pair posAA;
  nbsommet:=0;
  sommetdefiniA=sommetdefiniB=sommetdefiniC=false;
  rep_ad_A:=false;rep_ad_B:=false;rep_ad_C:=false;
  % on compte le nombre de sommets et on prend en compte lesquels sont définis
  forsuffixes i=t:
    nbsommet:=nbsommet+1;
    if nbsommet=1:
       if known i: sommetdefiniA:=true; rep_tri_AA:=i else: pair i fi
    elseif nbsommet=2:
       if known i: sommetdefiniB:=true; rep_tri_BB:=i else: pair i fi
    else:
       if known i: sommetdefiniC:=true; rep_tri_CC:=i else: pair i fi
    fi;
  endfor
  count_tmp:=0;
  posAA:=(0,0);anghh:=0;
  % on définit les valeurs de départ
  for i=s:
    count_tmp:=count_tmp+1;
    if count_tmp=1: rep_tri_lc:=i
    elseif count_tmp=2: rep_tri_angA:=i
    elseif count_tmp=3: rep_tri_angB:=i
    elseif pair i: if (not sommetdefiniA): posAA:=i fi
    else: anghh:=i
    fi;
  endfor
  % Angles droits ?
  if rep_tri_angA=90: rep_ad_A:=true elseif rep_tri_angB=90: rep_ad_B:=true elseif rep_tri_angA+rep_tri_angB=90: rep_ad_C:=true fi;
  rep_tri_angC:=180-rep_tri_angA-rep_tri_angB;
  count_tmp:=0;
  % on calcule les coordonnées des sommets
  forsuffixes i=t:
    count_tmp:=count_tmp+1;
    if count_tmp=1: %On définit A s'il n'existe pas. À partir de B ou C éventuellement
       if unknown i: 
          if sommetdefiniC and sommetdefiniB:
             i:=(rep_tri_lc,0) rotated (angle(rep_tri_CC-rep_tri_BB)+rep_tri_angB) shifted rep_tri_BB;
             rep_tri_angA:=(angle(rep_tri_CC-rep_tri_AA) mod 360)-(angle(rep_tri_BB-rep_tri_AA) mod 360);
          elseif sommetdefiniB: i:=(-rep_tri_lc,0) rotated anghh shifted rep_tri_BB
          elseif sommetdefiniC: 
             rep_tri_lb:=rep_tri_lc*abs(sind(rep_tri_angB)/sind(rep_tri_angC));
             i=(-rep_tri_lb,0) rotated (anghh+rep_tri_angA) shifted rep_tri_CC
          else: i:=posAA
          fi;
          rep_tri_AA:=i
       else:
          if sommetdefiniB: rep_tri_lc:= abs(rep_tri_BB-rep_tri_AA) fi;
          if sommetdefiniC: rep_tri_lb:= abs(rep_tri_CC-rep_tri_AA) fi;
          if sommetdefiniC and sommetdefiniB:
             rep_tri_angA:=(angle(rep_tri_CC-rep_tri_AA) mod 360)-(angle(rep_tri_BB-rep_tri_AA) mod 360);
             rep_tri_angB:=(angle(rep_tri_AA-rep_tri_BB) mod 360)-(angle(rep_tri_CC-rep_tri_BB) mod 360);
          fi;
       fi;
    elseif count_tmp=2: %On définit B à partir de C si C existe
       if unknown i:
          if sommetdefiniC: 
              i:= (rep_tri_lc,0) rotated (angle(rep_tri_CC-rep_tri_AA)-rep_tri_angA) shifted rep_tri_AA;
              rep_tri_angB:=(angle(rep_tri_AA-i) mod 360)-(angle(rep_tri_CC-i) mod 360);
          else:
              i:=(rep_tri_lc,0) rotated anghh shifted rep_tri_AA
          fi;
          rep_tri_BB:=i
       fi;
     else:           %On définit C s'il n'existe pas
        if unknown i:
           i = whatever[rep_tri_AA,rep_tri_BB rotatedaround(rep_tri_AA,rep_tri_angA)] = whatever[rep_tri_BB,rep_tri_AA rotatedaround(rep_tri_BB,-rep_tri_angB)];
        rep_tri_CC:=i
        fi;
     fi;
  endfor
  rep_tri_la:=abs(rep_tri_CC-rep_tri_BB);
  rep_tri_lb:=abs(rep_tri_CC-rep_tri_AA);
  rep_tri_angC:=180 - rep_tri_angB - rep_tri_angA;
enddef;


vardef defTriangleLLL(text t)(text s)=
  save nbsommet,count_tmp,posAA,anghh,sommetdefiniA,sommetdefiniB,sommetdefiniC,ab_tmp,or_tmp,ang_tmp;
  numeric nbsommet,count_tmp,anghh,ab_tmp,or_tmp,ang_tmp;
  boolean sommetdefiniA,sommetdefiniB,sommetdefiniC;
  pair posAA;
  nbsommet:=0;
  sommetdefiniA=sommetdefiniB=sommetdefiniC=false;
  rep_ad_A:=false;rep_ad_B:=false;rep_ad_C:=false;
  % on compte le nombre de sommets et on prend en compte lesquels sont définis
  forsuffixes i=t:
    nbsommet:=nbsommet+1;
    if nbsommet=1:
       if known i: sommetdefiniA:=true; rep_tri_AA:=i else: pair i fi
    elseif nbsommet=2:
       if known i: sommetdefiniB:=true; rep_tri_BB:=i else: pair i fi
    else:
       if known i: sommetdefiniC:=true; rep_tri_CC:=i else: pair i fi
    fi;
  endfor
  count_tmp:=0;
  posAA:=(0,0);anghh:=0;
  % on définit les valeurs de départ
  for i=s:
    count_tmp:=count_tmp+1;
    if count_tmp=1: rep_tri_lc:=i
    elseif count_tmp=2: rep_tri_lb:=i
    elseif count_tmp=3: rep_tri_la:=i
    elseif pair i: if (not sommetdefiniA): posAA:=i fi
    else: anghh:=i
    fi;
  endfor
  % Angles droits ?
  if arrondi(10**rep_int_arr,rep_tri_lc**2)=arrondi(10**rep_int_arr,(rep_tri_la)**2+(rep_tri_lb)**2):
     rep_ad_C:=true
  elseif arrondi(10**rep_int_arr,rep_tri_lb**2)=arrondi(10**rep_int_arr,(rep_tri_la)**2+(rep_tri_lc)**2):
       rep_ad_B:=true
  elseif arrondi(10**rep_int_arr,rep_tri_la**2)=arrondi(10**rep_int_arr,(rep_tri_lc)**2+(rep_tri_lb)**2):
         rep_ad_A:=true
  fi;
  count_tmp:=0;
  % on calcule les coordonnées des sommets
  forsuffixes i=t:
    count_tmp:=count_tmp+1;
    if count_tmp=1: %On définit A s'il n'existe pas. À partir de B ou C éventuellement
       if unknown i: 
          if sommetdefiniC and sommetdefiniB:
             rep_tri_la:=abs(rep_tri_BB-rep_tri_CC);
             ab_tmp:=((rep_tri_lc**2)+(rep_tri_la**2)-(rep_tri_lb**2))/(2*rep_tri_la);
             or_tmp:=sqrt((rep_tri_lc**2)-(ab_tmp**2));
             i := (ab_tmp,or_tmp) rotated angle(rep_tri_CC-rep_tri_BB) shifted rep_tri_BB;
          elseif sommetdefiniB: i:=(-rep_tri_lc,0) rotated anghh shifted rep_tri_BB
          elseif sommetdefiniC: 
             ang_tmp:=Arccosd(((rep_tri_lb**2)+(rep_tri_lc**2)-(rep_tri_la**2))/(2*rep_tri_lc*rep_tri_lb));
             i:=(-rep_tri_lb,0) rotated (ang_tmp + anghh) shifted rep_tri_CC
          else: i:=posAA
          fi;
          rep_tri_AA:=i
       else:
          if sommetdefiniB: rep_tri_lc:= abs(rep_tri_BB-rep_tri_AA) fi;
          if sommetdefiniC: rep_tri_lb:= abs(rep_tri_CC-rep_tri_AA) fi;
          if sommetdefiniC and sommetdefiniB:
             rep_tri_la:= abs(rep_tri_BB-rep_tri_CC)
          fi;
       fi;
    elseif count_tmp=2: %On définit B à partir de C si C existe
       if unknown i:
          if sommetdefiniC:
             rep_tri_lb:=abs(rep_tri_AA-rep_tri_CC);
             ab_tmp:=((rep_tri_la**2)+(rep_tri_lb**2)-(rep_tri_lc**2))/(2*rep_tri_lb);
             or_tmp:=sqrt((rep_tri_la**2)-(ab_tmp**2));
             i := (ab_tmp,or_tmp) rotated angle(rep_tri_AA-rep_tri_CC) shifted rep_tri_CC;
          else:
              i:=(rep_tri_lc,0) rotated anghh shifted rep_tri_AA
          fi;
          rep_tri_BB:=i
       fi;
     else:           %On définit C s'il n'existe pas
        if unknown i:
           ab_tmp:=((rep_tri_lb**2)+(rep_tri_lc**2)-(rep_tri_la**2))/(2*rep_tri_lc);
           or_tmp:=sqrt((rep_tri_lb**2)-(ab_tmp**2));
           i := (ab_tmp,or_tmp) rotated angle(rep_tri_BB-rep_tri_AA) shifted rep_tri_AA;
           rep_tri_CC:=i
        fi;
     fi;
  endfor
  rep_tri_angA:=(angle(rep_tri_CC-rep_tri_AA) -angle(rep_tri_BB-rep_tri_AA) ) mod 360;
  rep_tri_angB:=(angle(rep_tri_AA-rep_tri_BB)-angle(rep_tri_CC-rep_tri_BB)) mod 360;
  rep_tri_angC:=180 - rep_tri_angB - rep_tri_angA;
enddef;

vardef defTriangle(text q)=
  save count_tmp;
  numeric count_tmp;
  rep_ad_A:=false;rep_ad_B:=false;rep_ad_C:=false;
  count_tmp:=0;
  for i=q:
    count_tmp:=count_tmp+1;
    if count_tmp=1: rep_tri_AA:=i
    elseif count_tmp=2: rep_tri_BB:=i
    else: rep_tri_CC:=i
    fi;
  endfor
  rep_tri_lc:=abs(rep_tri_BB-rep_tri_AA);
  rep_tri_lb:=abs(rep_tri_CC-rep_tri_AA);
  rep_tri_la:=abs(rep_tri_BB-rep_tri_CC);
  rep_tri_angA:=abs(angle(rep_tri_CC-rep_tri_AA) -angle(rep_tri_BB-rep_tri_AA) );
  rep_tri_angB:=abs(angle(rep_tri_AA-rep_tri_BB)-angle(rep_tri_CC-rep_tri_BB));
  rep_tri_angC:=(180 - rep_tri_angB - rep_tri_angA);
  % Angles droits ?
  if arrondi(10**rep_int_arr,rep_tri_lc**2)=arrondi(10**rep_int_arr,(rep_tri_la)**2+(rep_tri_lb)**2):
     rep_ad_C:=true
  elseif arrondi(10**rep_int_arr,rep_tri_lb**2)=arrondi(10**rep_int_arr,(rep_tri_la)**2+(rep_tri_lc)**2):
       rep_ad_B:=true
  elseif arrondi(10**rep_int_arr,rep_tri_la**2)=arrondi(10**rep_int_arr,(rep_tri_lc)**2+(rep_tri_lb)**2):
         rep_ad_A:=true
  fi;
  rep_tri_lc:=arrondi(10**ArrondiCotes,rep_tri_lc);
  rep_tri_lb:=arrondi(10**ArrondiCotes,rep_tri_lb);
  rep_tri_la:=arrondi(10**ArrondiCotes,rep_tri_la);
enddef;

vardef TriangleLLA(text q)(text s)(text p)(text c)(text a)=
%% Macro utilisateur, definit le triangle et appelle la macro de tracé
  defTriangleLLA(q)(s);
  rep_type_triangle:=1;
  ProcessAffTriangles(q)(p)(c)(a)
enddef;

vardef triangleLLA(suffix AA,BB,CC)(text s)=
%% Macro utilisateur, definit le triangle et renvoie la ligne
  defTriangleLLA(AA,BB,CC)(s);
  rep_type_triangle:=0;
  triangle(AA,BB,CC)
enddef;

vardef TriangleLAA(text q)(text s)(text p)(text c)(text a)=
%% Macro utilisateur, definit le triangle et appelle la macro de tracé
  defTriangleLAA(q)(s);
  rep_type_triangle:=2;
  ProcessAffTriangles(q)(p)(c)(a)
enddef;

vardef triangleLAA(suffix AA,BB,CC)(text s)=
%% Macro utilisateur, definit le triangle et renvoie la ligne
  defTriangleLAA(AA,BB,CC)(s);
  rep_type_triangle:=0;
  triangle(AA,BB,CC)
enddef;

vardef TriangleLLL(text q)(text s)(text p)(text c)(text a)=
%% Macro utilisateur, definit le triangle et appelle la macro de tracé
  defTriangleLLL(q)(s);
  rep_type_triangle:=0;
  ProcessAffTriangles(q)(p)(c)(a)
enddef;

vardef triangleLLL(suffix AA,BB,CC)(text s)=
%% Macro utilisateur, definit le triangle et renvoie la ligne
  defTriangleLLL(AA,BB,CC)(s);
  rep_type_triangle:=0;
  triangle(AA,BB,CC)
enddef;


vardef Triangle(text q)(text p)(text c)(text a)=
  defTriangle(q);
  rep_type_triangle:=3;
  ProcessAffTriangles(q)(p)(c)(a)
enddef;



vardef ProcessAffTriangles(suffix A,B,C)(text p)(text c)(text a)=
%% Gestion de l'affichage 
  save chAA,chBB,chCC,cha,chb,chc,changA,changB,changC,count_tmp,ch,nang,AA,BB,CC;
  string chAA,chBB,chCC,cha,chb,chc,changA,changB,changC,ch;
  chAA=chBB=chCC=cha=chb=chc=changA=changB=changC="";
  pair AA,BB,CC;
  AA:=A;BB:=B;CC:=C;
  numeric count_tmp,nang;
  boolean tmpaffcotes,tmpaffangles;
  tmpaffcotes:=AffCotes;tmpaffangles:=AffAngles;
  if AffDonnees: AffCotes:=true;AffAngles:=true fi;
  % Noms des points passés en argument
   chAA:="$" & _chaine(A) & "$";
   chBB:="$" & _chaine(B) & "$";
   chCC:="$" & _chaine(C) & "$";
  % On modifie le nom des points si des valeurs sont passées dans p
  count_tmp:=0;
  for i=p:
    count_tmp:=count_tmp+1;
    if (count_tmp=1) and (i<>"~"):     chAA:=
    elseif (count_tmp=2) and (i<>"~"): chBB:=
    elseif (count_tmp=3) and (i<>"~"): chCC:=
    fi
    i;
  endfor
    % On modifie les noms si on est dans une transformation et on définit les points
    if  _ext_tr_<>"":
      chAA:=aj_extr_trsf(A,chAA);
      chBB:=aj_extr_trsf(B,chBB);
      chCC:=aj_extr_trsf(C,chCC);
    fi
  % On modifie les points si on est dans une transformation
    if _tr_bool_:
      def _transform_fig_= enddef;
      AA:=_transform_pt_(A);
      BB:=_transform_pt_(B);
      CC:=_transform_pt_(C);
    fi
  % Cotes données par les longueurs
    if AffCotes:
      if _tr_rapp_<>1:
        rep_tri_la:=rep_tri_la*_tr_rapp_;
        rep_tri_lb:=rep_tri_lb*_tr_rapp_;
        rep_tri_lc:=rep_tri_lc*_tr_rapp_;
      fi;
      if UnitCotes<>"":
        cha:=AffichageCoteAvecUnite(rep_tri_la,ArrondiCotes,UnitCotes);
        chb:=AffichageCoteAvecUnite(rep_tri_lb,ArrondiCotes,UnitCotes);
        chc:=AffichageCoteAvecUnite(rep_tri_lc,ArrondiCotes,UnitCotes);
      else:
        cha:=AffichageCoteSansUnite(rep_tri_la,ArrondiCotes);
        chb:=AffichageCoteSansUnite(rep_tri_lb,ArrondiCotes);
        chc:=AffichageCoteSansUnite(rep_tri_lc,ArrondiCotes);
      fi
    fi
  % On modifie les cotes si des valeurs sont passées dans c
  count_tmp:=0;
  for i=c:
    count_tmp:=count_tmp+1;
    if (count_tmp=1) and (i<>"~"):     cha:=
    elseif (count_tmp=2) and (i<>"~"): chb:=
    elseif (count_tmp=3) and (i<>"~"): chc:=
    fi
    i;
  endfor
  % Angles donnés ou calculés
  changA:=AffichageAngle(rep_tri_angA,ArrondiAngles);
  changB:=AffichageAngle(rep_tri_angB,ArrondiAngles);
  changC:=AffichageAngle(rep_tri_angC,ArrondiAngles);  
  % On modifie les angles si des valeurs sont passées dans a
  count_tmp:=0;
  for i=a:
    count_tmp:=count_tmp+1;
    if (count_tmp=1) and (i<>"~"):     changA:=
    elseif (count_tmp=2) and (i<>"~"): changB:=
    elseif (count_tmp=3) and (i<>"~"): changC:=
    fi
    i;
  endfor
  if count_tmp>0: nang:= count_tmp elseif AffAngles: nang:=3 else: nang:=0 fi;
  AffCotes:=tmpaffcotes;AffAngles:=tmpaffangles;
%
  TriangleComplet(AA,BB,CC)(chAA,chBB,chCC,chc,cha,chb,changA,changB,changC)
enddef;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%   TRANSFORMATIONS   %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

string extratransf;  % Chaine à ajouter aux nom des points 
boolean  _tr_bool_;    % Indique qu'il faudra traiter la chaine précédente
extratransf:="";
 _tr_bool_:=false;

numeric _tr_ang_rot_; % Angle à gérer pour l'affichage des points
_tr_ang_rot_:=0;
numeric _tr_rapp_; %Rapport d'agrandissement-réduction
_tr_rapp_:=1;


%% Translation

vardef translationpoint(expr u)(expr A)=
  A+u
enddef;

vardef translation(expr u)(text t)=
 type_tr:=1;
 if path t:
    t shifted u
 elseif pair t:
    t+u
 else:
   def _transform_pt_= translationpoint(u) enddef;
   def _transform_fig_= shifted u enddef;
   _ext_tr_:=extratransf;
    _tr_bool_ := true;
   t
   hide(_ext_tr_:=""; _tr_bool_:=false;
         def _transform_pt_=  enddef;
         def _transform_fig_= enddef;)
 fi
enddef;


%% Rotation

vardef rotationpoint(expr O,a)(expr A)=
  A rotatedaround(O,a)
enddef;

vardef rotation(expr O,a)(text t)=
 type_tr:=1;
 if path t:
    t rotatedaround(O,a)
 elseif pair t:
    t rotatedaround(O,a)
 else:
   def _transform_pt_= rotationpoint(O,a) enddef;
   def _transform_fig_= rotatedaround(_cart(O),a) enddef;
   _tr_ang_rot_:=a;
   _ext_tr_:=extratransf;
   _tr_bool_ := true;
   t _transform_fig_
   hide(_ext_tr_:="";_tr_bool_:=false;
        def _transform_pt_=  enddef;
        def _transform_fig_= enddef;
        _tr_ang_rot_:=0;)
 fi
enddef;

%% Symétrie centrale

vardef symetriecentralepoint(expr O)(expr A)=
  A rotatedaround(O,180)
enddef;

vardef symetriecentrale(expr O)(text t)=
 type_tr:=1;
 if path t:
    t rotatedaround(O,180)
 elseif pair t:
    t rotatedaround(O,180)
 else:
   def _transform_pt_= symetriecentralepoint(O) enddef;
   def _transform_fig_= rotatedaround(_cart(O),180) enddef;
   _tr_ang_rot_:=180;
   _ext_tr_:=extratransf;
   _tr_bool_ := true;
   t _transform_fig_
   hide(_ext_tr_:="";_tr_bool_:=false;
        def _transform_pt_=  enddef;
        def _transform_fig_= enddef;
        _tr_ang_rot_:=0;)
 fi
enddef;

%% Symétrie axiale

vardef symetrieaxialepoint(expr B,C)(expr A)=
  A reflectedabout(B,C)
enddef;

vardef symetrieaxiale(expr B,C)(text t)=
 type_tr:=1;
 if path t:
    t reflectedabout(B,C)
 elseif pair t:
    t reflectedabout(B,C)
 else:
   def _transform_pt_= symetrieaxialepoint(O) enddef;
   def _transform_fig_= reflectedabout(_cart(B),_cart(C)) enddef;
%   _tr_ang_rot_:=180;
   _ext_tr_:=extratransf;
   _tr_bool_ := true;
   t _transform_fig_
   hide(_ext_tr_:="";_tr_bool_:=false;
        def _transform_pt_=  enddef;
        def _transform_fig_= enddef;
        _tr_ang_rot_:=0;)
 fi
enddef;

%% Symétrie

vardef symetrie(text r)(text t)=
  if long_texte(r)=1:
    symetriecentrale(r)(t)
  else:
    symetrieaxiale(r)(t)
  fi
enddef;



%% Homothétie

vardef homothetiepoint(expr O,k)(expr A)=
  k[O,A]
enddef;

vardef homothetie(expr O,k)(text t)=
 type_tr:=1;
 if path t:
    t shifted -O scaled k shifted O
 elseif pair t:
    k[O,t]
 else:
    def _transform_pt_= homothetiepoint(O,k) enddef;
    def _transform_fig_= shifted -O scaled k shifted O enddef;
    _ext_tr_:=extratransf;
    _tr_bool_ := true;
    _tr_rapp_:=abs(k);
   t _transform_fig_
   hide(_ext_tr_:=""; _tr_bool_:=false;
         def _transform_pt_=  enddef;
         def _transform_fig_= enddef;
         _tr_rapp_:=1;)
 fi
enddef;

%% Similitude

vardef similitudepoint(expr O,k,a)(expr A)=
  k[O,A] rotatedaround(O,a)
enddef;

vardef similitude(expr O,k,a)(text t)=
 type_tr:=1;
 if path t:
    t rotatedaround(O,a) shifted -O scaled k shifted O
 elseif pair t:
    k[O,t] rotatedaround(O,a)
 else:
    def _transform_pt_= similitudepoint(O,k,a) enddef;
    def _transform_fig_=rotatedaround(_cart(O),a) shifted -O scaled k shifted O enddef;
    _ext_tr_:=extratransf;
    _tr_bool_ := true;
    _tr_rapp_:=abs(k);
   t _transform_fig_
   hide(_ext_tr_:=""; _tr_bool_:=false;
         def _transform_pt_=  enddef;
         def _transform_fig_= enddef;
         _tr_rapp_:=1;)
 fi
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   FONCTIONS USUELLES   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pi:=3.141592654;
e:=2.718281828;

vardef sin(expr x) = sind(180*x/pi) enddef;

vardef cos(expr x) = cosd(180*x/pi) enddef;

vardef tan(expr x) = sin(x)/cos(x) enddef;

vardef exp(expr x) = mexp(x*256) enddef;

vardef ln(expr x) = mlog(x)/256 enddef;

vardef ch(expr x)=(exp(x)+exp(-x))/2 enddef;

vardef sh(expr x)=(exp(x)-exp(-x))/2 enddef;

vardef Arctan(expr x)= pi*angle(1,x)/180 enddef;
vardef Arctand(expr x)= angle(1,x) enddef;

vardef Arcsin(expr x)= Arctan(x/sqrt(1-x**2)) enddef;
vardef Arcsind(expr x)= Arctand(x/sqrt(1-x**2)) enddef;

vardef Arccos(expr x)= pi/2-Arcsin(x) enddef;
vardef Arccosd(expr x)=  Arctand((sqrt(1-x**2))/x)enddef;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   REMPLISSAGE   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
let _fill_ = fill;

def fill expr c=
if path c: _fill_ c else: draw c fi
enddef;

numeric largbriques,hautbriques,decbriques;
largbriques=12;hautbriques=6;decbriques=6;

vardef briques(text t)=
 if long_texte(t)=0:
    repbriques(largbriques,hautbriques,decbriques)
 else:
    repbriques(t)
 fi
enddef;



vardef repbriques(expr larg,haut,dec)=
 numeric d;d=0;
 image(%
 for j=0 upto _hautcadre_/haut:
   olddraw ((0,0)--(_largcadre_,0)) shifted (0,j*haut);
   d:=d+dec;if d>larg: d:=d-larg fi;
   for i=0 upto _largcadre_/larg:
     olddraw ((0,0)--(0,haut)) shifted (i*larg+d,j*haut);
   endfor;
 endfor) shifted llcorner _cart(cadre)
enddef;


numeric pashach,anglehach;
pashach=5;anglehach=60;

vardef hachures(text t)=
 if long_texte(t)=0:
    rephachures(pashach,anglehach)
 else:
    rephachures(t)
 fi
enddef;

vardef rephachures(expr pas,angle)=
 image(%
 for i=floor(-_diag_/(2*pas)) upto ceiling(_diag_/(2*pas)):
 olddraw ((-_diag_/2,0)--(_diag_/2,0)) shifted (0,i*pas);
 endfor) rotated angle shifted center _cart(cadre)
enddef;

numeric pervagues,ampvagues,decvagues;
pervagues=20;ampvagues=3;decvagues=10;

vardef vagues(text t)=
 if long_texte(t)=0:
    repvagues(pervagues,ampvagues,decvagues)
 else:
    repvagues(t)
 fi
enddef;


vardef repvagues(expr per,amp,dec)=   %D'après le manuel de l'utilisateur de MetaPost
 image(%
 for j=0 upto _hautcadre_/dec:
   for i=0 upto _largcadre_/amp:
    olddraw ((0,0){curl 0}..(per/4,-amp)..(per/2,0){curl 0}..(3*per/4,amp)..(per,0){curl 0}) shifted (i*per,j*dec);
   endfor;
 endfor) shifted llcorner _cart(cadre)
enddef;


primarydef p avec c =
 begingroup
 save _des_;
 picture _des_;
 _des_:=c;
 clip _des_ to (p transformed _T);
 _des_
 endgroup
enddef;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%   DIVERS   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

string LaTeX_packages[],LaTeX_preamble,LaTeX_inputenc;
LaTeX_preamble:="";LaTeX_inputenc:="";

numeric n_packages;
n_packages:=0;

def LaTeX_AjoutPackage(expr t)=
     latexmp_packages:=latexmp_packages & "," & t
enddef;

def LaTeX_AjoutCommande(expr t)=
  LaTeX_preamble:=LaTeX_preamble  & t
enddef;

def LaTeX_Encodage(expr t)=
  LaTeX_inputenc:= t
enddef;



boolean LaTeXprem;
LaTeXprem:=true;

def LaTeXsetup =
   if LaTeXprem:
      LaTeXprem:=false;
      latexmp_preamble:=LaTeX_preamble;
      latexmp_inputenc:=LaTeX_inputenc;
      if latexmp_inputenc="": latexmp_inputenc:="utf8" fi;
      if latexmp_packages="": latexmp_packages:="fourier" fi;
      latexmp_packages:=latexmp_packages & ",textcomp,mathtools,mathrsfs,esvect[e],siunitx[output-decimal-marker={,}],babel[french]";
      latexmp_preamble:=latexmp_preamble & "\newcommand{\vect}[1]{\vv{#1}}" & "\newcommand{\calc}{\mathscr{C}}" & "\newcommand{\calp}{\mathscr{P}}" & "\DecimalMathComma" ;
   fi
enddef;

vardef LaTeX primary sc =   
  if known mplib: textext(sc) scaled defaultscale
  else: LaTeXsetup; textext(sc) scaled defaultscale
  fi
enddef;



%%%% Cotes et légendes


numeric dec_cote;
dec_cote:=4mm;
boolean traits_cote;
traits_cote:=false;
numeric angle_cote;
angle_cote:=-1;


% Mai 2017
% Légende avec flèche
vardef thelegende@#(expr lab,p) =
  image(%
  label@#(lab,point 0 of p);
  drawarrow p)
enddef;

def legende = draw thelegende enddef;

% Octobre 2021 - Juin 2022
% Cote sans flèche
vardef thecote@#(expr A, B, texte)=
  save I; pair I;
  save an,ac; numeric an,ac;
  I=0.5[A,B];
  an=angle(B-A);
  if angle_cote = -1: ac:=an else: ac:=angle_cote fi;
  if str @# ="top":
     thelabelangrot[an+90](texte,I,ac)
  else:
     thelabelangrot[an-90](texte,I,ac)
  fi
enddef;

def cote = draw thecote enddef;

% Mai 2022
% Cote sans flèche sans rotation du texte
vardef thecoted@#(expr A, B, texte)=
  save I; pair I;
  save an; numeric an;
  I=0.5[A,B];
  an=angle(B-A);
  if str @# ="top":
     thelabelang[an+90](texte,I)
  else:
     thelabelang[an-90](texte,I)
  fi
enddef;

def coted = draw thecoted enddef;

% Octobre 2021
% Cote avec flèche
vardef thecotefleche@#(expr a,b,t)=
  save pp,an;
  pair pp[];numeric an;
  an := if str @# ="top": 90 else: -90 fi;
  pp[1]:=dec_cote*(unitvector(b-a) rotated an) transformed inverse _T;
  pp[2]:=(dec_cote+1mm)*(unitvector(b-a) rotated an) transformed inverse _T;
  image(%
     if traits_cote:
        draw a--(a shifted pp[2]) dashed evenly;
        draw b--(b shifted pp[2]) dashed evenly;
     fi
     drawdblarrow ((a--b) shifted pp[1]);
     cote@#(a shifted pp[1],b shifted pp[1],t)
     )
enddef;

def cotefleche =
  draw thecotefleche
enddef;


%%%%%%%%%%%%%%%Figure intermédiaire%%%%%%%%%%%%%%%%%%%
def figureinter=
  clip currentpicture to (cadre transformed _T);
  shipit;
  _nfig:=_nfig+1;
  charcode:=charcode+1;
enddef;



def epaisseur=withpen pencircle scaled enddef;
def couleur=withcolor enddef;


vardef long_texte(text t)=
  save i;
  numeric i;
  i:=0;
  for n=t: i:=i+1; endfor
  i
enddef;

vardef _chaine(suffix s)= %met les termes en indice si le 2e est un chiffre
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

string _ext_tr_; % Chaine à rajouter aux étiquette lors d'une transformation
_ext_tr_:="";

vardef ajout_extra_trsf(expr ch)= % ajoute _ext_tr_ en dernière ou avant dernière pos
  save n;
  numeric n;
  n:=length(ch);
  if n>1:
    if substring (n-1,n) of ch = "$":
      substring (0,n-1) of ch & _ext_tr_ & substring (n-1,n) of ch
    else:
      ch & _ext_tr_
    fi
  else:
    ch
  fi
enddef;

vardef aj_extr_trsf(expr A,ch)=
  save tmpch,tmpchi,n;
  numeric n;
  string tmpch,tmpchi;
  n:=length(ch);
    if substring (n-1,n) of ch = "$":
      tmpchi:=substring (1,n-1) of ch; %   Sans les $
      if (ASCII _ext_tr_ > 47) and (ASCII _ext_tr_ < 58):
        tmpch:="$" & tmpchi & "_" & _ext_tr_ & "$";
        tmpchi:= tmpchi & "[]";
        scantokens("pair " & tmpchi & ";");
        tmpchi:= substring(0,n-1) of tmpchi & _ext_tr_ & "]";
      else:
        tmpchi:= tmpchi & _ext_tr_;
        tmpch:="$" & tmpchi & "$";
        scantokens("pair " & tmpchi & ";");
      fi
      scantokens(tmpchi) :=  _transform_pt_(A);
    else:
      tmpch:=ch;
    fi
  tmpch
enddef;


%%%%%%%%%%%%%%% MATHS %%%%%%%%%%%%%%%%%
vardef arrondidec(expr n,x)=  % arrondi de x à 10^-n
   save pp;
   numeric pp;
   pp:= 1  for i=1 upto n: *10 endfor;
   round(pp*x)/pp
enddef;


vardef arrondi(expr p,x)=              %arrondi de x au p-ième
%  round((x-floor(x))*p)/p+floor(x)
  round(p*x)/p
enddef;

vardef arrondimil(expr x)=
  arrondi(1000,x)
enddef;

vardef pgcd(expr a,b)=
  save r_,a_,b_;
  numeric r_,a_,b_;
  a_=a;b_=b;
  forever:
    r_:=a_ mod b_;
    exitif r_=0;
    a_:=b_;
    b_:=r_;
  endfor;
  b_
enddef;

vardef factorielle(expr n)=  %  Calcul de n!
  save $;
  numeric $;
  $:= 1  for i=1 upto n: * i endfor;
  $
enddef;

vardef binom(expr n,k)=    % calcul du coefficient binomial k parmi n
  save $,p;
  numeric $,p;
  p:= min(k,n-k);
  $:=1;
  for i=1 upto p:
   $:= ($*(n-i+1))/i;
  endfor
  $
enddef;

vardef binomiale(expr n,p,k)=
   save $,ktmp,ptmp;
   numeric $,ktmp,ptmp;
   if k>n-k:
      ktmp:=n-k;
      ptmp:=1-p
   else:
      ktmp:=k;
      ptmp:=p
   fi;
   $:=1;
   for i=1 upto ktmp:
      $:=((n-i+1)/i)*ptmp*$;
   endfor
   for i=1 upto n-ktmp:
      $:=$*(1-ptmp);
   endfor
   $
enddef;

% Mai 2017
% P(X=k) pour X suivant la loi de Poisson de paramètre lambda
vardef poisson(expr lambda,k)=
 exp(-lambda)
 for i=1 upto k:
   *(lambda/i)
 endfor
enddef;

%%%%%%%%%%%%%%% COULEURS %%%%%%%%%%%%%%%%%%%
cmykcolor Rouge,Vert,Bleu,Marron,Lime,Orange,Rose,Pourpre,Violet,Cyan,Magenta,Jaune,Olive,Noir,Grisfonce,Gris,Grisclair,Blanc,Beige,Marine,Moutarde,Vertfonce;

Rouge:=(0,1,1,0);
Vert:=(1,0,1,0);
Bleu:=(1,1,0,0);
Marron:=(0,0.53,1,0.65);
Lime:=(0.38,0,0.78,0.01);
Orange:=(0,0.5,1,0);
Rose:=(0,0.57,0.38,0.01);
Pourpre:=(0,0.91,0.59,0.38);
Violet:=(0.5,1,0,0.4);
Cyan:=(1,0,0,0);
Magenta:=(0,1,0,0);
Jaune:=(0,0,1,0);
Olive:=(0.21,0,0.75,0.45);
Noir:=(0,0,0,1);
Grisfonce:=(0,0,0,0.75);
Gris:=(0,0,0,0.5);
Grisclair:=(0,0,0,0.25);
Blanc:=(0,0,0,0);
Beige:=(0,0.14,0.37,0.22);
Marine:=(1,1,0,0.5);
Moutarde:=(0.04,0,1,0.19);
Vertfonce:=(1,0,1,0.5);


vardef conversion_cmyn_rvb(expr coul)=
  ((1-cyanpart coul)*(1-blackpart coul),(1-magentapart coul)*(1-blackpart coul),(1-yellowpart coul)*(1-blackpart coul))
enddef;

color rouge,vert,bleu,marron,lime,orange,rose,pourpre,violet,cyan,magenta,jaune,olive,noir,grisfonce,gris,grisclair,blanc,beige,marine,moutarde,vertfonce;

rouge:=conversion_cmyn_rvb(Rouge);
vert:=conversion_cmyn_rvb(Vert);
bleu:=conversion_cmyn_rvb(Bleu);
marron:=conversion_cmyn_rvb(Marron);
lime:=conversion_cmyn_rvb(Lime);
orange:=conversion_cmyn_rvb(Orange);
rose:=conversion_cmyn_rvb(Rose);
pourpre:=conversion_cmyn_rvb(Pourpre);
violet:=conversion_cmyn_rvb(Violet);
cyan:=conversion_cmyn_rvb(Cyan);
magenta:=conversion_cmyn_rvb(Magenta);
jaune:=conversion_cmyn_rvb(Jaune);
olive:=conversion_cmyn_rvb(Olive);
noir:=conversion_cmyn_rvb(Noir);
grisfonce:=conversion_cmyn_rvb(Grisfonce);
gris:=conversion_cmyn_rvb(Gris);
grisclair:=conversion_cmyn_rvb(Grisclair);
blanc:=conversion_cmyn_rvb(Blanc);
beige:=conversion_cmyn_rvb(Beige);
marine:=conversion_cmyn_rvb(Marine);
moutarde:=conversion_cmyn_rvb(Moutarde);
vertfonce:=conversion_cmyn_rvb(Vertfonce);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% TABLEAUX ET DAMIERS %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Octobre 2021
%

boolean rep_tab;
rep_tab:=false;

color table_coul[],grille_coul;
numeric grille_ep;



def tableau(text t)=
  save vv,ind;
  numeric vv[],ind;
  ind:=0;
  for i=t:
    ind:=ind +1;
    vv[ind]:=i;
  endfor
  vv[4]:=vv[ind];
  grille_coul:=black;
  grille_ep:=0.7;
  tab_init;
  deftableau(vv[1],vv[2],vv[3],vv[4]);
  table_coul[1]:=0.45white;
  table_coul[2]:=0.9white;
enddef;


def deftableau(expr n,p,u,v)=
  rep_tab:=true;
  repere(-1,n+2,u,-1,p+2,v);
  rep_tab_n:=n;rep_tab_p:=p;
enddef;


vardef grille(expr dx,dy)=
 save coultmp,eptmp;
 color coultmp; numeric eptmp;
 coultmp:=q_coul;eptmp:=q_ep;
 q_coul:=grille_coul;
 q_ep := grille_ep;
 save QuadXmin,QuadXmax,QuadYmin,QuadYmax;
 numeric QuadXmin,QuadXmax,QuadYmin,QuadYmax;
 QuadXmin:=0;QuadXmax:=rep_tab_n;QuadYmin:=0;QuadYmax:=rep_tab_p;
 quadrillage(dx,dy) shifted _cart(0.5,0.5) hide(q_coul:=coultmp;q_ep:=eptmp)
enddef;

vardef damier(expr dx,dy) text t=
 save coultmp,eptmp;
 color coultmp; numeric eptmp;
 coultmp:=q_coul;eptmp:=q_ep;
 q_coul:=grille_coul;
 q_ep := grille_ep;
 save count_tmp,pn;
 numeric count_tmp,pn;
 save QuadXmin,QuadXmax,QuadYmin,QuadYmax;
 numeric QuadXmin,QuadXmax,QuadYmin,QuadYmax;
 QuadXmin:=0;QuadXmax:=rep_tab_n;QuadYmin:=0;QuadYmax:=rep_tab_p;
 count_tmp:=2;pn=(((rep_tab_n-1) div dx)+1) mod 2;
 image(%
 for j=1 step dy until rep_tab_p:
   for i=1 step dx until rep_tab_n:
     count_tmp:=3-count_tmp;
     fill ((i,j) -- (i+dx,j) -- (i+dx,j+dy) -- (i,j+dy) -- cycle) shifted (-0.5,-0.5) withcolor table_coul[count_tmp];
   endfor
   if pn=0: count_tmp:=3-count_tmp fi;
 endfor
 draw quadrillage(dx,dy) shifted _cart(0.5,0.5) t;
 q_coul:=coultmp;q_ep:=eptmp
 )
enddef;


%%%%%%%%% Lignes particulières

vardef segmenth(expr x,y)=
  ((0,0)--(1,0)) shifted (x-0.5,y+0.5)
enddef;

vardef segmentv(expr x,y)=
  ((0,0)--(0,1)) shifted (x+0.5,y-0.5)
enddef;

vardef ligneh(expr m)(text t)=
  save ct;
  numeric ct;
  ct:=0;
  image(%
   for i=t:
     ct:=ct+1;
     if i=1:
       draw segmenth(ct,m)
     fi;
   endfor
  )
enddef;

vardef lignesh(text t)=
  save lig,col;
  numeric lig,col;
  lig:=rep_tab_p;col:=1;
  image(%
  for i=t:
    if i=1:
      draw segmenth(col,lig)
    fi;
    if col=rep_tab_n:
      lig:=lig-1;col:=1;
    else:
      col:=col+1;
    fi
  endfor
  )
enddef;

vardef lignesv(text t)=
  save lig,col;
  numeric lig,col;
  lig:=rep_tab_p;col:=0;
  image(%
  for i=t:
    if i=1:
      draw segmentv(col,lig)
    fi;
    if col=rep_tab_n:
      col:=0;lig:=lig-1;
    else:
      col:=col+1;
    fi
  endfor
  )
enddef;




% largeur d'une figure
vardef largeur_fig(expr pic)=
  abs(lrcorner pic - llcorner pic)
enddef;


% Coordonnées des cases du tableau %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% coordonnées dans le tableau
vardef _tab(expr x,y)=
  save px,py;
  numeric px,py;
  px := if inverse_coord_x: rep_tab_n + deb_coord_x - x else: x - deb_coord_x + 1 fi;
  py := if inverse_coord_y: rep_tab_p + deb_coord_y - y else: y - deb_coord_y + 1 fi;
  (px,py)
enddef;


def tab_init=        %initialisation variables
  % Style : "1", "i", "I", "a", "A"
  string style_coord_x,style_coord_y;
  style_coord_x:="1";
  style_coord_y:="1";

  % Première valeur
  numeric deb_coord_x,deb_coord_y;
  deb_coord_x:=1;
  deb_coord_y:=1;

  % Alignement
  string align_coord_x,align_coord_y;
  align_coord_x:="c";align_coord_y:="c";

  % Placement des coordonnées
  string place_coord;
  numeric place_coord_x,place_coord_y;
  place_coord:="bl";
  place_coord_x:=0;
  place_coord_y:=0;

  % Inversion coordonnées
  boolean inverse_coord_x,inverse_coord_y;
  inverse_coord_x:=false;
  inverse_coord_y:=false;
enddef;


vardef compoetiquette(expr n,t)=
  if t=1:     "$\num{" & decimal(n) & "}$"
  elseif t=2: "\strut\romannumeral" & decimal(n) 
  elseif t=3: "\MakeUppercase{\romannumeral" & decimal(n) & "}"
  elseif t=4: "\strut " & char(n+96)
  else:       char(n+64)
  fi
enddef;

vardef coordx=
  save stdeb,stfin,tt,aa;
  string stdeb,stfin;
  numeric tt,aa;
  if (substring (0,1) of place_coord = "h") or (substring (0,1) of place_coord = "t") or (substring (1,2) of place_coord = "h") or  (substring (1,2) of place_coord = "t"):
     place_coord_x:=1
  else:
     place_coord_x:=0
  fi;
  if style_coord_x="1":
     tt:=1
  elseif style_coord_x="i":
     tt:=2
  elseif style_coord_x="I":
     tt:=3
  elseif style_coord_x="a":
     tt:=4
  elseif style_coord_x="A":
     tt:=5
  fi;
  image(%
          for i=1 upto rep_tab_n:
            aa:= if inverse_coord_x: rep_tab_n+1-i else: i fi;
            if place_coord_x=0:
              label.bot(LaTeX(compoetiquette(i-1+deb_coord_x,tt)),(aa,0.5));
            else:
              label.top(LaTeX(compoetiquette(i-1+deb_coord_x,tt)),(aa,rep_tab_n+0.5));
            fi
          endfor
          )
enddef;


vardef coordy=
  save stdeb,stfin,tt,ll,cc;
  string stdeb,stfin;
  numeric tt,ll,cc;
  if align_coord_y="g": align_coord_y:="l" fi;
  if align_coord_y="d": align_coord_y:="r" fi;
  ll:=0;
  if (substring (0,1) of place_coord = "d") or (substring (0,1) of place_coord = "r") or (substring (1,2) of place_coord = "d") or  (substring (1,2) of place_coord = "r"):
     place_coord_y:=1
  else:
     place_coord_y:=0
  fi;
  if style_coord_y="1":
     tt:=1
  elseif style_coord_y="i":
     tt:=2
  elseif style_coord_y="I":
     tt:=3
  elseif style_coord_y="a":
     tt:=4
  elseif style_coord_y="A":
     tt:=5
  fi;
  for i=1 upto rep_tab_p:
       ll:=max(ll,largeur_fig(LaTeX(compoetiquette(i-1+deb_coord_y,tt))));
  endfor
  image(%
          for i=1 upto rep_tab_p:
            cc:= if inverse_coord_y: rep_tab_p+1-i else: i fi;
            if place_coord_y=0:
              label.lft(LaTeX("\makebox[" & decimal(ll/cm) & "cm][" & align_coord_y &"]{" &  compoetiquette(i-1+deb_coord_y,tt) & "}"),(0.5,cc));
            else:
              label.rt(LaTeX("\makebox[" & decimal(ll/cm) & "cm][" & align_coord_y &"]{" &  compoetiquette(i-1+deb_coord_y,tt) & "}"),(rep_tab_p+0.5,cc));
            fi
          endfor
          )
enddef;

vardef coord=
  image(%
   draw coordy;
   draw coordx;)
enddef;


%%%%%%%%% Numérotation dames

vardef numerotationdames=
 save ct;
 numeric ct;
 ct:=0;
 image(
  for j= rep_tab_p downto 1:
    for i=1 upto rep_tab_n:
     if (i+j) mod 2 = 0:
       ct:=ct+1;
       label("\bfseries" & decimal(ct),(i,j));
     fi
    endfor
  endfor
  )
enddef;

%%% Remplissage de cases %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vardef contour_case(expr n,m)=
  ((0,0)--(1,0)--(1,1)--(0,1)--cycle) shifted (_tab(n,m)-(0.5,0.5))
enddef;

vardef case(expr n,m) text t=
  image(%
    fill contour_case(n,m) t;
    draw (contour_case(n,m)) withpen pencircle scaled grille_ep withcolor grille_coul
    )
enddef;

vardef cases(text pts) text t=
  image(%
   for i= pts:
     draw case(xpart i,ypart i) t;
   endfor
  )
enddef;

%%% Placement d'un objet dans une case
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vardef thetablabel(expr s,n,m)=  % place l'objet s dans la case (n,m)
  save p; picture p;
  if picture s:  p=s
  elseif path s: p=image(draw s)
  else:          p = LaTeX(s)
  fi;
  thelabel(p,_tab(n,m))
enddef;

def tablabel = draw thetablabel enddef;

vardef tablabels(expr s)(text t) text cc=
  save li, co;
  numeric li,co;
  for i=t:
    if pair i:
      tablabel(s,xpart i,ypart i) cc;
    else:
      if rep_tab_p mod 2 = 1:
        li := rep_tab_p - ((2i-2) div rep_tab_n);
        if rep_tab_n mod 2 =0:
          co:=(2i-1 - (li mod 2)) mod rep_tab_n + 1;
        else:
          co:=(2i-2) mod rep_tab_n + 1;
        fi
      else:
        li := rep_tab_p - ((2i-1) div rep_tab_n);
        if rep_tab_n mod 2 =0:
          co:=(2i-1 - (li mod 2)) mod rep_tab_n + 1;
        else:
          co:=(2i-1) mod rep_tab_n + 1;
        fi
      fi
      tablabel(s,co,li);
    fi
  endfor
enddef;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%  PIXEL ART %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

color pixart_coul[],pixart_grille_coul;
numeric pixart_ep;
pixart_ep:=0.3;
pixart_grille_coul:=background;


def pixcoul(text t)=
 save cnt;
 numeric cnt;
 cnt:=0;
  for i=t:
    pixart_coul[cnt]:=i;
    cnt := cnt +1;
  endfor
enddef;

pixcoul(white,black,red,green,blue,cyan,magenta,jaune);

vardef contour_case_pix(expr n,m)=
  save tmpne,tmpse;
  pair tmpne,tmpse;
  tmpne:=0.5*(pixart_ep,pixart_ep) transformed inverse _T;
  tmpse:=0.5*(pixart_ep,-pixart_ep) transformed inverse _T;
  (((0,0)+tmpne)--((1,0)-tmpse)--((1,1)-tmpne)--((0,1)+tmpse)--cycle)
      shifted (_tab(n,m)-(0.5,0.5))
enddef;


vardef pixart(text t)=
 save _;
 numeric _;
 _:=-1;
 grille_ep:=pixart_ep;
 save cnt,lig,col;
 numeric cnt,lig,col;
 cnt:=0;
 lig:=rep_tab_p;col:=1;
 image(%
   for i=t:
   if numeric i:
    if i>-1: filldraw contour_case_pix(col,lig) withcolor pixart_coul[i] fi
   elseif color i:
     filldraw contour_case_pix(col,lig) withcolor i
   fi;
    if col=rep_tab_n:
      lig:=lig-1;col:=1;
    else:
      col:=col+1;
    fi
   endfor
 )
enddef;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%  DESSINS %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% pions - 2 versions pioni et pionii
vardef Pioni(expr coulint,coulext)=
  image(%
         fill cercle((0,0),0.4) couleur coulint;
         draw cercle((0,0),0.4) epaisseur (0.5*Ux/cm) couleur coulext;
         draw cercle((0,0),0.25) epaisseur (0.5*Ux/cm) couleur coulext;
         draw (-0.05,0)--(0.05,0) epaisseur (0.5*Ux/cm) couleur coulext;
         draw (0,0.05)--(0,-0.05) epaisseur (0.5*Ux/cm) couleur coulext
  )
enddef;


%vardef Pionnoiri = Pioni(noir,0.85blanc) enddef;
%vardef Pionblanci = Pioni(0.9blanc,noir) enddef;


vardef Pionii(expr coulint,coulext)=
  image(%
    fill (cercle((0,0),0.4) yscaled 0.5) couleur coulint;
    draw (arccercle((-0.4,0),(0,0),(0.4,0)) yscaled 0.5) epaisseur (0.5*Ux/cm) couleur coulext;
    fill (cercle((0,0),0.4) yscaled 0.5 shifted (0,0.1)) couleur coulint;
    draw (cercle((0,0),0.4) yscaled 0.5 shifted (0,0.1)) epaisseur (0.5*Ux/cm) couleur coulext;
    draw (-0.4,0)--(-0.4,0.1) epaisseur (0.5*Ux/cm) couleur coulext;
    draw (0.4,0)--(0.4,0.1) epaisseur (0.5*Ux/cm) couleur coulext;
  )
enddef;

% dame - 2 versions damei et dameii
vardef Damei(expr coulint,coulext)=
  image(%
         fill cercle((0,0),0.4) couleur coulint;
         draw cercle((0,0),0.4) epaisseur (0.5*Ux/cm) couleur coulext;
         fill cercle((0,0),0.08) couleur coulext;
         draw (-0.15,0)--(0.15,0) epaisseur (0.9*Ux/cm) couleur coulext;
         draw (0,0.15)--(0,-0.15) epaisseur (0.9*Ux/cm) couleur coulext;
         draw (-0.1,0.1)--(0.1,-0.1) epaisseur (0.3*Ux/cm) couleur coulext;
         draw (0.1,0.1)--(-0.1,-0.1) epaisseur (0.3*Ux/cm) couleur coulext
  )
enddef;

%vardef Damenoirei = Dame(noir,0.85blanc) enddef;
%vardef Dameblanchei = Dame(0.9blanc,noir) enddef;

vardef Dameii(expr coulint,coulext)=
  image(%
    fill (cercle((0,0),0.4) yscaled 0.5) couleur coulint;
    draw (arccercle((-0.4,0),(0,0),(0.4,0)) yscaled 0.5) epaisseur (0.5*Ux/cm) couleur coulext;
    fill (cercle((0,0),0.4) yscaled 0.5 shifted (0,0.1)) couleur coulint;
    draw (arccercle((-0.4,0),(0,0),(0.4,0)) yscaled 0.5 shifted (0,0.1)) epaisseur (0.5*Ux/cm) couleur coulext;
    fill (cercle((0,0),0.4) yscaled 0.5 shifted (0,0.2)) couleur coulint;
    draw (cercle((0,0),0.4) yscaled 0.5 shifted (0,0.2)) epaisseur (0.5*Ux/cm) couleur coulext;
    draw (-0.4,0)--(-0.4,0.2) epaisseur (0.5*Ux/cm) couleur coulext;
    draw (0.4,0)--(0.4,0.2) epaisseur (0.5*Ux/cm) couleur coulext;
  )
enddef;


def Pion(expr v)=
  if v=1:
     Pioni
  else:
    Pionii
  fi
enddef;

def Dame(expr v)=
  if v=1:
     Damei
  else:
    Dameii
  fi
enddef;

% Croix
vardef Croix=
  interim linecap:=butt;
  image(%
  draw (0,0)--(1,1) epaisseur (1.2*Ux/cm);
  draw (0,1)--(1,0) epaisseur (1.2*Ux/cm)
  )
enddef;

% Trou
vardef Trou=
  image(%
   fill fullcircle scaled 0.8)
enddef;

% Marques - 2 versions
vardef Marquei(expr coul)=
  save tmp;
  save a,b,c,d;
  pair a,b,c,d;
  picture tmp;
  tmp:=image(
    fill fullcircle scaled 0.8 yscaled 0.4 shifted (0,-0.2) couleur coul);
  a=llcorner tmp;b= lrcorner tmp; c=urcorner tmp; d=ulcorner tmp;
  setbounds tmp to (a--b--(2.2c-1.2b)--(2.2d-1.2a)--cycle);
  tmp
enddef;

vardef Marqueii(expr coul)=
  image(%
    fill((0.1,0.1)--(0.9,0.1)--(0.9,0.9)--(0.1,0.9)--cycle) couleur coul;
  )
enddef;

def Marque(expr v)=
  if v=1:
    Marquei
  else:
    Marqueii
  fi
enddef;

% Mur
vardef Mur (expr coulint,coulext)=
  image(%
  fill ((0,0)--(1,0)--(1,1)--(0,1)--cycle) couleur coulint;
  draw ((0,0)--(0.3,0)--(0.3,0.7)--(0,0.7)--cycle) epaisseur (Ux/cm) couleur coulext;
  draw ((0.3,0)--(1,0)--(1,0.3)--(0.3,0.3)--cycle) epaisseur (Ux/cm) couleur coulext;
  draw ((1,0.3)--(1,1)--(0.7,1)--(0.7,0.3)--cycle) epaisseur (Ux/cm) couleur coulext;
  draw ((0.7,1)--(0,1)--(0,0.7)--(0.7,0.7)--cycle) epaisseur (Ux/cm) couleur coulext;
  )
enddef;

% Plot
vardef Plot(expr couli,coulii)=
  image(%
   fill cercle((0,0),0.4) yscaled 0.4 couleur coulii;
   fill cercle((0,0),0.3) yscaled 0.4 couleur couli;
   fill (-0.3,0)--(-0.05,0.6)--(0.05,0.6)--(0.3,0)--cycle couleur couli;
   fill cercle((0,0),0.05) yscaled 0.4 shifted (0,0.6) couleur coulii;
  )
enddef;

% Caisse
vardef Caisse(expr coulint,coulext)=
  save a,b;
  pair a,b,c,d;
  a=(0.1,0.1);b=(0.9,0.1);
  c=(0.9,0.9);d=(0.1,0.9);
  image(%
  fill (a--b--c--d--cycle) couleur coulint;
  for i=1 upto 7:
    draw (a--b) shifted (0,0.1*i) epaisseur (0.7*Ux/cm) couleur coulext;
  endfor
  draw a--b--c--d--cycle epaisseur (1.5*Ux/cm) couleur coulext;
  draw a--c epaisseur (1.5*Ux/cm) couleur coulext;
  draw b--d epaisseur (1.5*Ux/cm) couleur coulext;
  )
enddef;



% Robot
vardef Rob_chaine=
  (1.5,1)..(1.7,0.7)..(2.4,0.4)..(4.4,0.3)..(6.4,0.3)..(7,0.5)..(7,1.2)..(6.2,2.2)..(5.3,3)..(3.9,3.5)..(3.1,3.6)..(2.3,3.2)..(1.7,2.3)..cycle
enddef;


vardef Rob_corps=
  (6.28,7.33)..(7.23,6.67)..(7.5,6.36)..(7.6,5)..(7.5,3.53)..(7.35,3.23)..(7,3)..(6.43,2.82)..(5.95,2.74)..(2,3.27)..(1.64,3.98)..(1.46,5.11)..(1.46,6.41)..(1.99,7.34)..cycle
enddef;

vardef Rob_tete=
  fullcircle xscaled 5.9 yscaled 3.8 shifted (4.1,8.8)
enddef;

vardef Rob_haut=
  fullcircle xscaled 5.5 yscaled 1.6 shifted (4.1,10.4)
enddef;

vardef Rob_bras=
 (3,5.5)..(3.4,5)..(3.8,4.5)..(4.2,4.1)..(5,4)..(6,4)..(6.3,4.8)..(5.9,5.4)..(4.9,5.2)..(4.9,5.7)..(4.6,6.2)..(4.1,6.5)..(3.5,6.3)..(3,5.9)..cycle
enddef;

vardef Rob_oeild=
  fullcircle scaled 0.9 shifted (5.1,8.7)
enddef;

vardef Rob_oeilg=
  fullcircle xscaled 0.6 yscaled 0.85 shifted (6.67,9.1)
enddef;

vardef Rob_bouche=
  (5.9,7.8)..(6.2,7.8)..(6.4,7.9)
enddef;

vardef Robotdroite(expr couli,coulii)=
  save tmp;
  picture tmp;
  tmp:=image(%
        fill Rob_chaine shifted (2,2) withcolor couli;
        draw Rob_chaine shifted (2,2) withcolor 0.7couli+0.3white epaisseur (8*Ux/cm);
        fill Rob_corps withcolor coulii;
        fill Rob_tete withcolor couli;
        fill buildcycle(Rob_tete,Rob_haut) withcolor coulii;
        fill Rob_oeild withcolor coulii;
        fill Rob_oeilg withcolor coulii;
        draw Rob_bouche withcolor coulii epaisseur (5*Ux/cm);
        fill Rob_bras withcolor 0.7couli+0.3white;
        fill Rob_chaine withcolor couli;
        draw Rob_chaine withcolor 0.7couli+0.3white epaisseur (8*Ux/cm)) scaled 0.085;
  tmp
enddef;

vardef Robotgauche(expr couli,coulii)=
   Robotdroite(couli,coulii) xscaled -1
enddef;


% Bateau
vardef Bat_coque=
  (-0.8,0.4)--(-0.55,0.1)--(0.8,0.1)--(0.7,0.25)--(0.1,0.25)--(-0.1,0.4)--cycle
enddef;

vardef Bat_cabine=
  (-0.6,0.4)--(-0.5,0.55)--(0.2,0.55)--(0.3,0.4)--(0.5,0.4)--(0.6,0.25)--(0.1,0.25)--(-0.1,0.4)--cycle
enddef;

vardef Bat_barre=
  (-0.54,0.475)--(-0.15,0.475)
enddef;

vardef Bat_cheminee=
  (0,0)--(0,0.1)--(0.15,0.1)--(0.15,0)--cycle
enddef;


vardef Bateau(expr num, coul)=
  save e,fx,fy;
  numeric e,fx,fy;
  fx := if     num=2: 1
        elseif num=1: 0.5
        elseif num=3: 1.6
        elseif num=4: 2.3
        else: 1
        fi;
  fy := if     num=2: 1
        elseif num=1: 0.7
        elseif num=3: 1.2
        elseif num=4: 1.3
        else: 1
        fi;
  e:=(0.7*fx-0.15*num)/(num+1);
  image(%
   fill Bat_coque xscaled fx yscaled fy couleur (0.75*blanc+0.25*coul) ;
   fill Bat_cabine xscaled fx yscaled fy couleur blanc;
   for i=0 upto num-1:
     fill Bat_cheminee shifted ((-0.5,0.55) xscaled fx yscaled fy) shifted (e+i*(e+0.15),0)  couleur (0.75*blanc+0.25*coul);
     draw Bat_cheminee shifted ((-0.5,0.55) xscaled fx yscaled fy) shifted (e+i*(e+0.15),0)  epaisseur (0.7*Ux/cm) couleur coul;
   endfor
   draw Bat_coque xscaled fx yscaled fy epaisseur (0.7*Ux/cm) couleur coul;
   draw Bat_cabine xscaled fx yscaled fy epaisseur (0.7*Ux/cm) couleur coul;
   draw Bat_barre xscaled fx yscaled fy epaisseur (1.2*Ux/cm) couleur coul;
  )
enddef;

% Eau
vardef Eau(expr coul)=
  save vague;
  path vague;
  vague :=(-0.3,0)..(-0.15,0.05){right}..(0,0)..(0.15,-0.05){right}..(0.3,0);
  image(%
   draw vague epaisseur (Ux/cm) couleur coul;
   draw vague shifted (0,0.15) epaisseur (Ux/cm) couleur coul;
   )
enddef;

% Flèche
vardef Fleche(expr coulint,coulext)=
  save tmp;
  path tmp;
  tmp:= ((0.45,0)--(0.45,3)--(1,3)--(0,4)--(-1,3)--(-0.45,3)--(-0.45,0)--cycle) scaled 0.2;
  image(fill tmp couleur coulint;
        draw tmp couleur coulext)
enddef;

%%%%%%% Anglicisation
let coordxy = repere;
let endxy = fin;
let xaxis = axex;
let yaxis = axey;
let axis = axes;
let interaxis = interaxes;
let setlab = setval;
let arrowaxis = flecheaxe;
let frame = cadre;


let purple = pourpre;

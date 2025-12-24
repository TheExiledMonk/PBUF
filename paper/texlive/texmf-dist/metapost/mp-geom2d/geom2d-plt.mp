%
% Courbe param�tr�e
% ------------------------------------------------------------------------------
vardef Courbe(suffix fx)(suffix fy)(expr ti,tf,n) =
    save fpas; fpas := (tf-ti)/n;
    (fx(ti),fy(ti)) for i=1 upto n: ..(fx(ti+i*fpas),fy(ti+i*fpas)) endfor
enddef;

vardef Curve(suffix fx)(suffix fy)(expr ti,tf,n) = Courbe(fx)(fy)(ti,tf,n) enddef;

%
% Repr�sentation de fonction
% ------------------------------------------------------------------------------
vardef Representation(suffix f)(expr ti,tf,n) =
    save fpas;
    fpas := (tf-ti)/n;
    (ti,f(ti)) for i=1 upto n: ..(ti+i*fpas,f(ti+i*fpas)) endfor
enddef;

vardef Plot(suffix f)(expr ti,tf,n) = Representation(f)(ti,tf,n) enddef;

%
% Courbe en polaire
% ------------------------------------------------------------------------------
vardef CourbeEnPolaires(suffix r)(expr ti,tf,n) =
    save fpas,t;
    fpas := (tf-ti)/n;
    r(ti)*(cos(ti),sin(ti))
    for i=1 upto n: hide(t:=ti+i*fpas) .. r(t)*(cos(t),sin(t)) endfor
enddef;

vardef PolarCurve(suffix r)(expr ti,tf,n) = CourbeEnPolaires(r)(ti,tf,n) enddef;

%
% Champs de vecteurs
% ------------------------------------------------------------------------------
vardef ChampVecteurs(suffix f)(expr x,y,px,py,dx,couleur) =
  for i = 0 upto (x - repXmin)/px:
     for j = 0 upto (y - repYmin)/py:
       drawarrow
        (((0,0)--dx*unitvector((1,f(x-i*px,y-j*py))))
                 shifted (x-i*px,y-j*py)) gddEnPlace
                 withcolor couleur;
     endfor
     for j = 0 upto (repYmax - y)/py:
       drawarrow
        (((0,0)--dx*unitvector((1,f(x-i*px,y+j*py))))
                 shifted (x-i*px,y+j*py)) gddEnPlace
                 withcolor couleur;
     endfor
  endfor
  for i = 0 upto (repXmax - x)/px:
     for j = 0 upto (y - repYmin)/py:
       drawarrow
        (((0,0)--dx*unitvector((1,f(x+i*px,y-j*py))))
                 shifted (x+i*px,y-j*px)) gddEnPlace
                 withcolor couleur;
     endfor
     for j = 0 upto (repYmax - y)/py:
       drawarrow
        (((0,0)--dx*unitvector((1,f(x+i*px,y+j*py))))
                 shifted (x+i*px,y+j*py)) gddEnPlace
                 withcolor couleur;
     endfor
  endfor
enddef;

vardef VectorField(suffix f)(expr x,y,px,py,dx,couleur) =
ChampVecteurs(f)(x,y,px,py,dx,couleur) enddef;

%
% Champs de vecteurs f R^2 -> R^2
% ------------------------------------------------------------------------------
vardef ChampVecteursDD(suffix f)(expr x,y,px,py,dx,couleur) =
  for i = 0 upto (x - repXmin)/px:
     for j = 0 upto (y - repYmin)/py:
       drawarrow
        (((0,0)--dx*unitvector(f(x-i*px,y-j*py)))
                 shifted (x-i*px,y-j*py)) gddEnPlace
                 withcolor couleur;
     endfor
     for j = 0 upto (repYmax - y)/py:
       drawarrow
        (((0,0)--dx*unitvector(f(x-i*px,y+j*py)))
                 shifted (x-i*px,y+j*py)) gddEnPlace
                 withcolor couleur;
     endfor
  endfor
  for i = 0 upto (repXmax - x)/px:
     for j = 0 upto (y - repYmin)/py:
       drawarrow
        (((0,0)--dx*unitvector(f(x+i*px,y-j*py)))
                 shifted (x+i*px,y-j*px)) gddEnPlace
                 withcolor couleur;
     endfor
     for j = 0 upto (repYmax - y)/py:
       drawarrow
        (((0,0)--dx*unitvector(f(x+i*px,y+j*py)))
                 shifted (x+i*px,y+j*py)) gddEnPlace
                 withcolor couleur;
     endfor
  endfor
enddef;

vardef VectorFieldDD(suffix f)(expr x,y,px,py,dx,couleur) =
ChampVecteursDD(f)(x,y,px,py,dx,couleur) enddef;

endinput;

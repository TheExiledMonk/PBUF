% mpkiviat.mp v0.1
% @Author: Maxime Chupin
% @Date: 22 may 2024
% @License: LATEX Project Public License 1.3c. 

if not known mplib: input latexmp fi;
pair _axis[];

numeric _axis_number;
numeric _kiviat_unit;

numeric _lattice_unit,_lattice_max,_value_max,_unit_value;
_lattice_unit := 1;
_lattice_max :=10;
_value_max := 10;
_unit_value := 1;
_kiviat_unit:=0.3cm;


pair _kiviat_lines[][];
path _kiviat_path[];
color _kiviat_lines_color[];
numeric _line_nbr;
_line_nbr:=0;

def set_lattice_grid(expr u,m)=
    save _nbr;
    _nbr := (m/u);
    _value_max := m;
    _unit_value := u;
    _lattice_max := floor(_nbr);
enddef;


string _axis_names[];

color _kiviat_lattice_color, _kiviat_axis_color;
_kiviat_axis_color:= black;
_kiviat_lattice_color:= 0.8*white;

vardef set_axis_color(expr c)=
    _kiviat_axis_color:= c;
enddef;

def set_kiviat_unit(expr u)= 
    _kiviat_unit:=u;
enddef;

vardef set_lattice_color(expr c)=
    _kiviat_lattice_color:= c;
enddef;

vardef set_axis(text t)=
    _line_nbr:=0;
    save _st,_i;
    _axis_number:=0;
    for _st=t:
        _axis_number:= _axis_number+1;
        _axis_names[_axis_number]:=_st;
    endfor
    for i:=1 upto _axis_number:
        _axis[i]:=(cosd(i/_axis_number*360)*_lattice_max*_kiviat_unit,sind(i/_axis_number*360)*_lattice_max*_kiviat_unit);
    endfor
enddef;





boolean _legend;
_legend:=true;

% function to determine the point of the bb that intersect the line 
% determined by the angle of the corresponding axis
vardef _pointbb (expr p, a)=
    save _bb,_d,_point;
    path _bb,_d;
    pair _point; 
    _bb := bbox p;
    _d := (0,0)--(3000*cosd(a),3000*sind(a));
    _point := _bb intersectionpoint _d;
    _point
enddef;

% set legends true/false
vardef set_axis_legends(expr b)=
    _legend:=b;
enddef;

def draw_axis =
    save _plabel;
    picture _plabel;
    for i:=_lattice_unit step _lattice_unit until _lattice_max :
        draw for j:=1 upto _axis_number: (i/_lattice_max)*_axis[j]--endfor cycle withcolor
        _kiviat_lattice_color;
    endfor
    for i:=1 upto _axis_number:
        drawarrow (0,0)--1.2*_axis[i] withcolor _kiviat_axis_color;
    endfor
    if(_legend):
        for i:=1 upto _axis_number:
            _plabel := image(label(textext(_axis_names[i]),origin)); % passage
            % par image et label pour centrer la bbox
            draw _plabel  
             shifted (1.2*_axis[i]+labeloffset*(cosd(angle _axis[i]),sind(angle
             _axis[i]))+_pointbb(_plabel,angle _axis[i])) ;
        endfor;
    fi
enddef;

def draw_grad(expr p,s)(expr j)=
    for i:=1 upto _lattice_max:
        unfill bbox (textext(p&decimal(i*_unit_value)&s) shifted (i/_lattice_max*_axis[j]));
        draw textext(p&decimal(i*_unit_value)&s) shifted (i/_lattice_max*_axis[j]);
    endfor
enddef;



vardef _build_line(text t)=
    save _values,_i;
    _line_nbr:=_line_nbr+1;
    _i:=0;
    for _values=t:
        _i:=_i+1;
        _kiviat_lines[_line_nbr][_i]:=(_values/_value_max)*_axis[_i];
        if(_i=1):
            _kiviat_path[_line_nbr]:=_kiviat_lines[_line_nbr][_i];
        else:
            _kiviat_path[_line_nbr]:=_kiviat_path[_line_nbr]--_kiviat_lines[_line_nbr][_i];
        fi
    endfor
    _kiviat_path[_line_nbr]:=_kiviat_path[_line_nbr]--cycle;
enddef;


boolean _line_mark_b;
_line_mark_b:=true;
numeric _line_mark_scale;
_line_mark_scale:=1;

def set_line_mark_scale(expr n)=
    _line_mark_scale:=n;
enddef;

def set_line_mark(expr b)=
    _line_mark_b:=b;
enddef;

def line_mark_square(expr p)=
    ((((-1,-1)--(1,-1)--(1,1)--(-1,1)--cycle) scaled _line_mark_scale) shifted p)
enddef;

def line_mark_circle(expr p)=
    ((fullcircle scaled 2.5 scaled _line_mark_scale) shifted p)
enddef;

string _mark_type;
_mark_type := "square";
string _save_type[];
boolean _save_mark_b[];
numeric _save_scale[];

def set_line_mark_type(expr s)= 
    _mark_type:=s;
enddef;

def _mark_line(expr c)=
    _save_type[_line_nbr]:=_mark_type;
    _save_scale[_line_nbr]:=_line_mark_scale;
    for i:=1 upto _axis_number:
        fill 
        if(_mark_type="square"):
            line_mark_square
        elseif(_mark_type="circle"):
            line_mark_circle
        elseif(_mark_type="custom"):
            line_mark_custom
        fi
        (_kiviat_lines[_line_nbr][i]) withcolor c;
    endfor;
enddef;

vardef draw_line(text t)(expr c)=
    _build_line(t);
    _kiviat_lines_color[_line_nbr]:=c;
    draw _kiviat_path[_line_nbr] withcolor c;
    _save_mark_b[_line_nbr]:=_line_mark_b;
    if(_line_mark_b):
        _mark_line(c);
    fi;
enddef;


vardef filldraw_line(text t)(expr c)=
    _build_line(t);
    _kiviat_lines_color[_line_nbr]:=c;
    alphafill(_kiviat_path[_line_nbr],c,0.1);
    draw _kiviat_path[_line_nbr] withcolor c;
    _save_mark_b[_line_nbr]:=_line_mark_b;
    if(_line_mark_b):
        _mark_line(c);
    fi;
enddef;

vardef _anchor@# (expr p)=
    if(str @# = "top"):
    (1/2*(urcorner p)+1/2*(ulcorner p))
    elseif(str @# = "bot"):
    (1/2*(lrcorner p)+1/2*(llcorner p))
    elseif(str @# = "lft"):
    (1/2*(llcorner p)+1/2*(ulcorner p))
    elseif(str @# = "rt" ):
    (1/2*(lrcorner p)+1/2*(urcorner p))
    elseif(str @# = "urt"):
    (urcorner p)
    elseif(str @# = "ulft"):
    (ulcorner p)
    elseif(str @# = "lrt"):
    (lrcorner p)
    elseif(str @# = "llft"):
    (llcorner p)
    elseif(str @# = "c"):
    (1/4*(lrcorner p)+1/4*(urcorner p)+1/4*(llcorner p)+1/4*(ulcorner p))
    else:
    (1/4*(lrcorner p)+1/4*(urcorner p)+1/4*(llcorner p)+1/4*(ulcorner p))
    fi  

enddef;

vardef draw_legends@# (text l) =
    save _mypic,i,_s_scale;
    _s_scale := _line_mark_scale;
    picture _mypic;
    _mypic = image(
        i:=0;
        for s=l:
        i:=i+1;
        draw ((0,0)--(0.7*_kiviat_unit,0)) shifted (0,i*(_kiviat_unit)) withcolor
        _kiviat_lines_color[i];
        if(_save_mark_b[i]):
            _line_mark_scale:=_save_scale[i];
             fill 
            if(_save_type[i]="square"):
            line_mark_square
            elseif(_save_type[i]="circle"):
            line_mark_circle
            elseif(_save_type[i]="custom"):
            line_mark_custom
            fi
            (0.5[(0,0),(0.7*_kiviat_unit,0)] )shifted (0,i*(_kiviat_unit))
            withcolor _kiviat_lines_color[i];
        fi
        label.rt(textext(s),(0.7*_kiviat_unit,0) shifted (0,i*(_kiviat_unit)));
        endfor
        draw bbox currentpicture;
        _line_mark_scale := _s_scale ;
    );
    draw _mypic 
    if(str @# = "top"):
    shifted (_anchor.top(currentpicture)-_anchor.bot(_mypic))
    elseif(str @# = "bot"):
    shifted (_anchor.bot(currentpicture)-_anchor.top(_mypic))
    elseif(str @# = "lft"):
    shifted (_anchor.lft(currentpicture)-_anchor.rt(_mypic))
    elseif(str @# = "rt" ):
    shifted (_anchor.rt(currentpicture)-_anchor.lft(_mypic))
    elseif(str @# = "urt"):
    shifted (_anchor.urt(currentpicture)-_anchor.ulft(_mypic))
    elseif(str @# = "ulft"):
    shifted (_anchor.ulft(currentpicture)-_anchor.urt(_mypic))
    elseif(str @# = "lrt"):
    shifted (_anchor.lrt(currentpicture)-_anchor.llft(_mypic))
    elseif(str @# = "llft"):
    shifted (_anchor.llft(currentpicture)-_anchor.lrt(_mypic))
    else:
    shifted (_anchor.rt(currentpicture)-_anchor.lft(_mypic))
    fi
    ;
enddef;
% transparency Phan : http://www-math.univ-poitiers.fr/~phan/metalpha.html
% revisité par Jean-Michel Sarlat
color   _tc;
numeric _ta;
picture _ap;

def _AlphaFill expr c =
    _ap := nullpicture;
    _AlphaPicture(currentpicture,c);
    addto currentpicture also _ap;
enddef;

vardef _AlphaPicture(expr p,c) =
    save p_; picture p_;
    p_ := nullpicture;
    addto p_ contour c withcolor _ta[background,_tc];
    for p__ within p:
      if (not clipped p__) and (not bounded p__):
        addto p_ also p__ withcolor
         _ta[(redpart p__,greenpart p__,bluepart p__), _tc];
      else:
        begingroup save _ap;
         picture _ap; _ap = nullpicture;
         _AlphaPicture(p__,pathpart p__);
         addto p_ also _ap;
        endgroup;
      fi
    endfor;
    clip p_ to c;
    addto _ap also p_;
enddef;

vardef alphafill(expr p,c,a)  =
    _tc := c; _ta := a;
    _AlphaFill  p
enddef;

endinput;
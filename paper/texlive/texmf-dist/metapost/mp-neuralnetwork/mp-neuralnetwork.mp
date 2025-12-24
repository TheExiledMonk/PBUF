% Don't load this package twice:
if known neuralnetwork_version: expandafter endinput; fi;

numeric neuralnetwork_version;string neuralnetwork_date;
neuralnetwork_version=0.1;
neuralnetwork_date="2025/01/27";
% The banner:
message "******* mp-neuralnetwork " & decimal (neuralnetwork_version) &
        " (c) Maxime Chupin (" & neuralnetwork_date & ") *******";message "";

input metaobj;
input mp-neuralnetwork-metaobj-patch;
if not known mplib: input latexmp fi;
input colorbrewer-rgb;

interim ahangle := 40;
interim ahlength := 2;

numeric _ANN_Vspace,_ANN_Hspace;
_ANN_Vspace:=1.5cm;
_ANN_Hspace:=2.2cm;



boolean
_ANN_Hidden_b,_ANN_Input_b,_ANN_Output_b,_ANN_Legends_b,_ANN_show_node_labels_b,_ANN_weights_b,_ANN_abstract_b,_ANN_Legends_b,_ANN_colored_b,_ANN_user_color;

string _ANN_input_legend,_ANN_output_legend, _ANN_hidden_legend;
string _ANN_weights;

numeric _ANN_instance_nbr;
numeric _ANN_label_nbr;
color _ANN_colors[],_ANN_connect_color;



% init of colors
for ii:=0 upto 30:
    _ANN_colors[ii]:=white;
endfor;

_ANN_connect_color := 0.4*white;

_ANN_input_legend := "Input layer";
_ANN_output_legend := "Output layer";
_ANN_hidden_legend := "Hidden layer(s)";

_ANN_colored_b := true;
_ANN_Legends_b:=false;
_ANN_abstract_b:=false;
_ANN_Hidden_b := false;
_ANN_Input_b := false;
_ANN_Output_b := false;
_ANN_show_node_labels_b := true;
_ANN_weights_b := false;
_ANN_user_color:=false;
_ANN_weights := "w";

def mpneural_init_values=
    _ANN_Vspace:=1.5cm;
    _ANN_Hspace:=2.2cm;
    _ANN_input_legend := "Input layer";
    _ANN_output_legend := "Output layer";
    _ANN_hidden_legend := "Hidden layer(s)";

    _ANN_colored_b := true;
    _ANN_Legends_b:=false;
    _ANN_abstract_b:=false;
    _ANN_Hidden_b := false;
    _ANN_Input_b := false;
    _ANN_Output_b := false;
    _ANN_show_node_labels_b := true;
    _ANN_weights_b := false;
    _ANN_user_color:=false;
    _ANN_weights := "w";
    _ANN_connect_color := 0.4*white;
enddef;




def set_vspace(expr d)=
    _ANN_Vspace:=d;
enddef;

def set_hspace(expr d)=
    _ANN_Hspace:=d;
enddef;
def set_connection_color(expr c)=
    _ANN_connect_color:=c;
enddef;


def show_node_labels(expr b)=
    _ANN_show_node_labels_b:=b;
enddef;

def framed_input(expr b)=
    _ANN_Input_b:=b;
enddef;

def framed_output(expr b)=
    _ANN_Output_b:=b;
enddef;

def framed_hidden(expr b)=
    _ANN_Hidden_b:=b;
enddef;

def print_legends(expr b)= 
    _ANN_Legends_b:=b;
enddef;

def colored_layers(expr b)=
    _ANN_colored_b:=b;
enddef;

vardef set_layer_colors(text t)=
    save ii,_i_layer;
    % initialisation 
    % init of colors
    for ii:=1 upto 30:
        _ANN_colors[ii]:=(1,1,1);
    endfor;
    for c=t:
        show(c);
        if(numeric c):
            _i_layer:=c;
        elseif(color c):
            _ANN_colors[_i_layer]:=c;
        fi
    endfor;
    _ANN_user_color:=true;
enddef;

def print_weights(expr b)=
    _ANN_weights_b:=b;
enddef;

def set_weights_notation(expr s)=
    _ANN_weights:=s;
enddef;


def set_input_legend(expr s)=
    _ANN_input_legend := s;
enddef;

def set_output_legend(expr s)=
    _ANN_output_legend := s;
enddef;

def set_hidden_legend(expr s)=
    _ANN_hidden_legend := s;
enddef;


_ANN_instance_nbr :=0;

def drawANN(text t) text options=
    % t : list of ("str1",int1,"str2",int2, etc.)

    save prefixANN,_ANN_i,_ANN_i_couple,_ANN_n,_ANN_lb,_ANN_lb_int;
    string prefixANN;

    numeric _ANN_n[],_ANN_i,_ANN_i_couple;
    string _ANN_lb[],_ANN_lb_int[];

    _ANN_instance_nbr := _ANN_instance_nbr+1;

    prefixANN:="ANN"&decimal(_ANN_instance_nbr);
    _ANN_abstract_b:=false;
    _ANN_i:=1;
    _ANN_i_couple:=1;
    % parsing the list
    for x=t:
        if(odd _ANN_i):
            _ANN_lb[_ANN_i_couple]:=prefixANN&"_"&x;
        else:
            if(numeric x):
                _ANN_n[_ANN_i_couple]:=x;
                _ANN_i_couple:=_ANN_i_couple+1;
            else:
                _ANN_abstract_b:=true;
                _ANN_n[_ANN_i_couple]:=5;
                _ANN_lb_int[_ANN_i_couple]:=x;
                _ANN_i_couple:=_ANN_i_couple+1;
            fi
        fi
        _ANN_i:=_ANN_i+1;
    endfor

    % colors 
    if(_ANN_user_color=false):
        for i:=1 upto _ANN_i_couple-1:
            _ANN_colors[i]:=Spectral[min(11,_ANN_i_couple-1)][i mod 11];
        endfor
    fi

    if(_ANN_abstract_b=true): % small size for input and output
        _ANN_n[1]:=4;
        _ANN_n[_ANN_i_couple-1]:=4;
    fi
    for i:=1 upto _ANN_i_couple -1:
        for j:=1 upto _ANN_n[i]:
            if( not _ANN_show_node_labels_b):
                newCircle.scantokens(_ANN_lb[i])[j](textext("$\phantom{a}$"))
                if(_ANN_colored_b): "filled(true)","fillcolor("&str _ANN_colors[i] &")", fi options;
            elseif(_ANN_abstract_b): % abstract
             
                if(j<3): % for the two first ones
                    newCircle.scantokens(_ANN_lb[i])[j](textext("$"&substring
                    (4+length(decimal(_ANN_instance_nbr)),length _ANN_lb[i]) of
                    _ANN_lb[i]&"_{"&decimal(j)&"}$")) if(_ANN_colored_b): "filled(true)","fillcolor("&str _ANN_colors[i] &")", fi options;
                elseif(j=3): % dots
                    newBox.scantokens(_ANN_lb[i])[j](image(drawoptions(withpen
                    pencircle scaled 1bp);drawdot (0,0);drawdot
                    (0,0.1cm);drawdot
                    (0,0.2cm);drawoptions();)) if(_ANN_colored_b):
                 "filled(true)","fillcolor("&str _ANN_colors[i] &")", fi ;
                else: % lasts
                  
                    if((i=1) or (i=_ANN_i_couple-1)): %if input or output only last
                        newCircle.scantokens(_ANN_lb[i])[j](textext("$"&substring
                        (4+length(decimal(_ANN_instance_nbr)),length _ANN_lb[i]) of
                        _ANN_lb[i]&"_{"&_ANN_lb_int[i]&"}$")) if(_ANN_colored_b): "filled(true)","fillcolor("&str _ANN_colors[i] &")", fi options;
                    else:
                        if(j=4):
                            newBox.scantokens(_ANN_lb[i])[j](image(drawoptions(withpen
                    pencircle scaled 1bp);drawdot (0,0);drawdot
                    (0,0.1cm);drawdot (0,0.2cm);drawoptions();)) if(_ANN_colored_b): "filled(true)","fillcolor("&str _ANN_colors[i] &")", fi ;
                        else:
                            newCircle.scantokens(_ANN_lb[i])[j](textext("$"&substring
                            (4+length(decimal(_ANN_instance_nbr)),length _ANN_lb[i]) of
                            _ANN_lb[i]&"_{"&_ANN_lb_int[i]&"}$")) if(_ANN_colored_b): "filled(true)","fillcolor("&str _ANN_colors[i] &")", fi  options;
                        fi
                    fi
                fi
            else:
                newCircle.scantokens(_ANN_lb[i])[j](textext("$"&substring
                (4+length(decimal(_ANN_instance_nbr)),length _ANN_lb[i]) of
                _ANN_lb[i]&"_{"&decimal(j)&"}$")) if(_ANN_colored_b):
                "filled(true)","fillcolor("&str _ANN_colors[i] &")", fi options;
            fi
            if(j=1):
                if(i=1):
                    scantokens(_ANN_lb[i])[j].c = origin;
                else:
                    scantokens(_ANN_lb[i])[j].c = scantokens(_ANN_lb[i-1])[j].c+(_ANN_Hspace,0)+(0,-(_ANN_n[i]-_ANN_n[i-1])/3.5*_ANN_Vspace);
                fi
            else:
                scantokens(_ANN_lb[i])[j].n-scantokens(_ANN_lb[i])[j-1].s=(0,_ANN_Vspace);
            fi
        endfor
        for j:=1 upto _ANN_n[i]:
            drawObj(scantokens(_ANN_lb[i])[j]);
        endfor
    endfor
    % group for hiddenlayer
    
    if(_ANN_i_couple-1>3):
        newGroup.scantokens(prefixANN)_hidden( for i:=2 upto (_ANN_i_couple-2): for j:=1 upto _ANN_n[i]: 
        scantokens(_ANN_lb[i])[j] if((i<_ANN_i_couple-2)or(j<_ANN_n[i])): , fi
        endfor endfor) "framed(true)","dx(3pt)","dy(3pt)";
    fi
    if(_ANN_Hidden_b): 
        drawObj(scantokens(prefixANN)_hidden) ;
    fi
    % rectangle for inputlayer
    
    if(_ANN_n[1]>1):
        newGroup.scantokens(prefixANN)_input(for j:=1 upto _ANN_n[1]: 
        scantokens(_ANN_lb[1])[j] if(j<_ANN_n[1]): , fi
        endfor)
        "framed(true)","dx(3pt)","dy(3pt)";
    else:
        newBox.scantokens(prefixANN)_input(scantokens(_ANN_lb[1])[1]) "filled(false)","framed(true)","dx(3pt)","dy(3pt)";
    fi
    if(_ANN_Input_b): 
        drawObj(scantokens(prefixANN)_input);
    fi

    % rectangle for outputlayer
    
    if(_ANN_n[_ANN_i_couple-1]>1):
        newGroup.scantokens(prefixANN)_output(for j:=1 upto _ANN_n[_ANN_i_couple-1]: 
        scantokens(_ANN_lb[_ANN_i_couple-1])[j] if(j<_ANN_n[_ANN_i_couple-1]): , fi
        endfor)
        "framed(true)","dx(3pt)","dy(3pt)";
    else:
        newBox.scantokens(prefixANN)_output(scantokens(_ANN_lb[_ANN_i_couple-1])[1]) "filled(false)","framed(true)","dx(3pt)","dy(3pt)";
    fi
    if(_ANN_Output_b): 
        drawObj(scantokens(prefixANN)_output);
    fi

    % links
    for i:=1 upto _ANN_i_couple-1:
        if(i>1):
            for j:=1 upto _ANN_n[i-1]:
                for k:=1 upto _ANN_n[i]:
                    ncline(scantokens(_ANN_lb[i-1])[j])(scantokens(_ANN_lb[i])[k])
                    "linecolor(_ANN_connect_color)"
                    ;
                endfor
            endfor

        fi
    endfor
    % weights
    if(_ANN_weights_b):
        for j:=1 upto _ANN_i_couple-2:
            newBox.scantokens(prefixANN)_weights[j](textext("$"&_ANN_weights&"_{ij}^{("&decimal(j)&")}$"));
            scantokens(prefixANN)_weights.[j].c=1/4*(scantokens(_ANN_lb[j])[1].c+scantokens(_ANN_lb[j])[_ANN_n[j]].c+scantokens(_ANN_lb[j+1])[1].c+scantokens(_ANN_lb[j+1])[_ANN_n[j+1]].c);
            drawObj(scantokens(prefixANN)_weights[j]);
        endfor;
    fi
    % labels
    if(_ANN_Legends_b):
        newBox.scantokens(prefixANN)_legend_input(textext(_ANN_input_legend)) "framed(false)";
        scantokens(prefixANN)_legend_input.n=scantokens(prefixANN)_input.s;
        drawObj(scantokens(prefixANN)_legend_input);
        newBox.scantokens(prefixANN)_legend_output(textext(_ANN_output_legend)) "framed(false)";
        scantokens(prefixANN)_legend_output.n=scantokens(prefixANN)_output.s;
        drawObj(scantokens(prefixANN)_legend_output);

        if(_ANN_i_couple-1>3):
            newBox.scantokens(prefixANN)_legend_hidden(textext(_ANN_hidden_legend)) "framed(false)";
            scantokens(prefixANN)_legend_hidden.n=scantokens(prefixANN)_hidden.s;
            drawObj(scantokens(prefixANN)_legend_hidden);
        fi
    fi
    % build global group
    newGroup.scantokens(prefixANN)_global(for i:=1 upto (_ANN_i_couple-1): for j:=1 upto _ANN_n[i]: 
        scantokens(_ANN_lb[i])[j] if((i<_ANN_i_couple-1)or(j<_ANN_n[i])): , fi
        endfor endfor);
enddef;


_ANN_label_nbr:=0;
vardef labelANN(expr s)(expr a,b) text options=
    % s: string for the label
    % (a,b): point at which put the label in relative coordinate, that is
    % $(0,0)$ at bottom left of the bounding box of the complete graph and
    % $(1,1)$ at top right.  

    save _xmin, _xmax, _ymin,_ymax;
    save prefixANN;
    string prefixANN;
    _ANN_label_nbr:=_ANN_label_nbr+1;

    prefixANN:="ANN"&decimal(_ANN_instance_nbr);
    (_xmin,_ymin)=scantokens(prefixANN)_global.sw;
    (_xmax,_ymax)=scantokens(prefixANN)_global.ne;
    newBox.scantokens(prefixANN)_label scantokens(decimal(_ANN_label_nbr))(textext(s))  options;
    scantokens(prefixANN)_label
    scantokens(decimal(_ANN_label_nbr)).c=(_xmin+a*(_xmax-_xmin),_ymin+b*(_ymax-_ymin));
    drawObj(scantokens(prefixANN)_label scantokens(decimal(_ANN_label_nbr)));
enddef;

endinput
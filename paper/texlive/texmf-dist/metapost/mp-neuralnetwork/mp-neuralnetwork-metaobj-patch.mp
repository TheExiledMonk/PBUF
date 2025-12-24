%=====================================================================
% |Group| class (DR, 4 December 2024, after idea by Maximum Chupin)
%                based on the |HBox| class

% |Group|: Generic Group
% |@#| is a name for an object (must be a suffix)
% |@#| will be the number of the object, but will also be used
% as a prefix for other variables.
vardef newGroup@#(text sublist) text options =
  ExecuteOptions(@#)(options);
  assignObj(@#,"Group");
  StandardInterface;
  save n,i,eq;numeric n,i;string eq;
  n=0;
  forsuffixes $:=sublist:n:=n+1;endfor;
  ObjSubArray(sb)(n); % |n| is the number of elements
  ObjNumeric nst,leftmax,rightmax,botmax,topmax;
  setNumeric(nst)(n);
  i=0;
  forsuffixes $:=sublist:i:=i+1;
    SubObjectOfArray(sb[i],$);
  endfor;
  setMaxBB@#;
  % we now build the equations:
  % horizontal equation: horizontal space at left and right
  eq="xpart(obj(@#sb[@#leftmax]).w-@#w)" &
    "=xpart(@#e-obj(@#sb[@#rightmax]).e)=" &
       decimal(OptionValue@#("dx")) & ";";
  % vertical equation: vertical space at top and bottom
  eq:= eq & "ypart(obj(@#sb[@#botmax]).s-@#s)" &
           "=ypart(@#n-obj(@#sb[@#topmax]).n)=" &
       decimal(OptionValue@#("dy")) & ";";
  ObjCode StandardEquations,eq;
  StandardTies;
enddef;

vardef setMaxBB@#=
  @#leftmax:=1; % initialization of indexes
  @#rightmax:=1;
  @#botmax:=1;
  @#topmax:=1;
  for i:=2 upto @#nst:
    if xpart(obj(@#sb[i]).w-obj(@#sb[@#leftmax]).w)<0:@#leftmax:=i;fi;
    if xpart(obj(@#sb[i]).e-obj(@#sb[@#rightmax]).e)>0:@#rightmax:=i;fi;
    if ypart(obj(@#sb[i]).s-obj(@#sb[@#botmax]).s)<0:@#botmax:=i;fi;
    if ypart(obj(@#sb[i]).n-obj(@#sb[@#topmax]).n)>0:@#topmax:=i;fi;    
  endfor;
enddef;

streamline("Group")("(text sublist)","suffixlist(sublist)");

def BpathGroup(suffix n)= StandardBpath(n) enddef;

def drawGroup(suffix n)=
  drawFramedOrFilledObject_(n);
  drawObjArray(n)(sb);
  drawMemorizedPaths_(n);
enddef;

% Default values of |Group|:
setObjectDefaultOption("Group")("dx")(0mm);
setObjectDefaultOption("Group")("dy")(0mm);
setObjectDefaultOption("Group")("framed")(false);
setObjectDefaultOption("Group")("filled")(false);
setObjectDefaultOption("Group")("fillcolor")(black);
setObjectDefaultOption("Group")("framewidth")(.5bp);
setObjectDefaultOption("Group")("framecolor")(black);
setObjectDefaultOption("Group")("framestyle")("");
setObjectDefaultOption("Group")("shadow")(false); % no shadow by default
setObjectDefaultOption("Group")("shadowcolor")(black);
%---------------------------------------------------------------------------


endinput
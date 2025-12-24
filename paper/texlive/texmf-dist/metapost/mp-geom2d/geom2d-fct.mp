numeric Pi;
numeric _E;

Pi := 3.14159265;
_E  := 2.71828183;

vardef sin(expr x) = sind(x/Pi*180) enddef;
vardef cos(expr x) = cosd(x/Pi*180) enddef;
vardef tan(expr x) = sin(x)/cos(x) enddef;

vardef exp(expr x) = _E**x enddef;

vardef ln(expr x) = mlog(x)/256 enddef;
vardef ch(expr x) = (_E**x + _E**(-x))/2 enddef;
vardef cosh(expr x) = ch(x) enddef;
vardef sh(expr x) = (_E**x - _E**(-x))/2 enddef;
vardef sinh(expr x) = sh(x) enddef;
vardef th(expr x) = (_E**(2*x) - 1)/(_E**(2*x) + 1) enddef;
vardef tanh(expr x) = th(x) enddef;

vardef arcsin(expr t) = angle((sqrt(1-t**2),t)) / 180 * Pi enddef;
vardef arccos(expr t) = angle(t,(sqrt(1-t**2))) / 180 * Pi enddef;
vardef arctan(expr t) = angle(1,t) / 180 * Pi enddef;

endinput
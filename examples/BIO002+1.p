%------------------------------------------------------------------------------
% File     : BIO002+1 : TPTP v7.3.0. Bugfixed v6.4.1.
% Domain   : Biology
% Problem  : A cell has a part
% Version  : [Wes13] axioms : Especial.
% English  :

% Refs     : [CE+14] Chaudri et al. (2014), Comparative Analysis of Knowled
%          : [Wes13] Wessel (2013), Email to G. Sutcliffe
% Source   : [TPTP]
% Names    :

% Status   : Theorem
% Rating   : 0.93 v7.3.0, 0.83 v7.0.0
% Syntax   : Number of formulae    : 9162 ( 716 unit)
%            Number of atoms       : 374192 (121152 equality)
%            Maximal formula depth : 4682 (  42 average)
%            Number of connectives : 395266 (30236   ~;  46   |;356538   &)
%                                         (   0 <=>;8446  =>;   0  <=;   0 <~>)
%                                         (   0  ~|;   0  ~&)
%            Number of predicates  : 7015 (   0 propositional; 1-4 arity)
%            Number of functors    : 73905 (30403 constant; 0-1 arity)
%            Number of variables   : 11119 (   0 sgn;11118   !;   1   ?)
%            Maximal term depth    :    3 (   2 average)
% SPC      : FOF_THM_RFO_SEQ

% Comments : 
% Bugfixes : v6.4.1 - Double quoted numbers in BIO001+0.ax
%------------------------------------------------------------------------------
include('Axioms/BIO001+0.ax').
%------------------------------------------------------------------------------
fof(a_cell,axiom,(
    cell_1(a_cell) )).

fof(ask,conjecture,(
    ? [Y] : has_part_2(a_cell,Y) )).

%------------------------------------------------------------------------------

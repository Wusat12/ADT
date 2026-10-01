#(ly:set-option 'crop #t)
\version "2.24.4"
\header { 
tagline = " "
}

\drums {
    \numericTimeSignature
    \time 4/4
    <<{
        \partial 2 \repeat unfold 8 cymrb8 \repeat unfold 4 {cymrb32 hh32} sn32 hh32 \repeat unfold 3 {cymrb32 hh32}
    }
    \\ {
        \stemUp s4 sn4 s4 sn4 s4
    }
    \\ {
        \stemDown \repeat unfold 48 bd32
    }>>
}
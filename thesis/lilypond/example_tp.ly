#(ly:set-option 'crop #t)
\version "2.24.4"
\header { 
tagline = " "
}

\drums {
    \numericTimeSignature
    \time 4/4
    <<{
        \repeat unfold 4 hh8
        \repeat unfold 2 hh4
    }
    \\ {
        bd4 sn4 r4 sn4
    }>>
}
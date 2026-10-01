#(ly:set-option 'crop #t)
\version "2.24.4"
\header { 
tagline = " "
}
\paper {
    line-width = 150\mm 
}

\drums {
    \numericTimeSignature
    \time 4/4
    <<{
        \partial 8*7 <<cymrb8 hh8>>\repeat unfold 5 cymrb8 {cymrb64 cymrb64 r32*3} | \repeat unfold 6 cymrb8 s4
    }
    \\ {
        r8 sn8 bd16 sn16 r16 sn16 bd8 sn8 bd8 bd8 bd8 sn8 sn8 r16 sn16 bd8 sn16 sn16 bd8
    }>>
}

#ifndef _GPU_PSA_FUNCTION_
#define _GPU_PSA_FUNCTION_
#include "all.cuh"
#include <stdlib.h>  /* for malloc, free */
#include <string.h>  /* for memcmp, memmove */

namespace STEMMERGPU {



__forceinline__ __device__ int32_t cons(char* zb, int32_t& zk, int32_t& zj, int32_t i)
{  switch (zb[i])
   {  case 'a': case 'e': case 'i': case 'o': case 'u': return false;
      case 'y': return (i == 0) ? true : !cons(zb, zk, zj, i - 1);
      default: return true;
   }
}

__forceinline__ __device__ int32_t m(char* zb, int32_t& zk, int32_t& zj)
{  int32_t n = 0;
   int32_t i = 0;
   int32_t j = zj;
   while(true)
   {  if (i > j) return n;
      if (! cons(zb, zk, zj, i)) break; i++;
   }
   i++;
   while(true)
   {  while(true)
      {  if (i > j) return n;
            if (cons(zb, zk, zj, i)) break;
            i++;
      }
      i++;
      n++;
      while(true)
      {  if (i > j) return n;
         if (! cons(zb, zk, zj, i)) break;
         i++;
      }
      i++;
   }
}

__forceinline__ __device__ int32_t vowelinstem(char* zb, int32_t& zk, int32_t& zj)
{
   int32_t j = zj;
   int32_t i; for (i = 0; i <= j; i++) if (! cons(zb, zk, zj, i)) return true;
   return false;
}

__forceinline__ __device__ int32_t doublec(char* zb, int32_t& zk, int32_t& zj, int32_t j)
{
   char * b = zb;
   if (j < 1) return false;
   if (b[j] != b[j - 1]) return false;
   return cons(zb, zk, zj, j);
}

__forceinline__ __device__ int32_t cvc(char* zb, int32_t& zk, int32_t& zj, int32_t i)
{  if (i < 2 || !cons(zb, zk, zj, i) || cons(zb, zk, zj, i - 1) || !cons(zb, zk, zj, i - 2)) return false;
   {  int32_t ch = zb[i];
      if (ch  == 'w' || ch == 'x' || ch == 'y') return false;
   }
   return true;
}

__forceinline__ __device__ int32_t ends(char* zb, int32_t& zk, int32_t& zj, const char * s)
{  int32_t length = s[0];
   char * b = zb;
   int32_t k = zk;
   if (s[length] != b[k]) return false; /* tiny speed-up */
   if (length > k + 1) return false;
   for(int32_t i = 0; i < length; ++i) {
       if(b[ k - length + 1 + i] != s[i + 1]) {
           return false;
       }
   }
   //if (memcmp(b + k - length + 1, s + 1, length) != 0) return false;
   zj = k-length;
   return true;
}


__forceinline__ __device__ void setto(char* zb, int32_t& zk, int32_t& zj, const char * s)
{  int32_t length = s[0];
   int32_t j = zj;
   for(int32_t i = 0; i < length; ++i)
   {
       zb[i + j + 1] = s[i + 1];
   }
   //memmove(zb + j + 1, s + 1, length);
   zk = j+length;
}

__forceinline__ __device__ void r(char* zb, int32_t& zk, int32_t& zj, const char * s) { 
    if (m(zb, zk, zj) > 0) 
        setto(zb, zk, zj, s);
}

__forceinline__ __device__ void step1ab(char* zb, int32_t& zk, int32_t& zj)
{
   char * b = zb;
   if (b[zk] == 's')
   {  if (ends(zb, zk, zj, "\04" "sses")) zk -= 2; else
      if (ends(zb, zk, zj, "\03" "ies")) setto(zb, zk, zj, "\01" "i"); else
      if (b[zk - 1] != 's') zk--;
   }
   if (ends(zb, zk, zj, "\03" "eed")) { if (m(zb, zk, zj) > 0) zk--; } else
   if ((ends(zb, zk, zj, "\02" "ed") || ends(zb, zk, zj, "\03" "ing")) && vowelinstem(zb, zk, zj))
   {  zk = zj;
      if (ends(zb, zk, zj, "\02" "at")) setto(zb, zk, zj, "\03" "ate"); else
      if (ends(zb, zk, zj, "\02" "bl")) setto(zb, zk, zj, "\03" "ble"); else
      if (ends(zb, zk, zj, "\02" "iz")) setto(zb, zk, zj, "\03" "ize"); else
      if (doublec(zb, zk, zj, zk))
      {  zk--;
         {  int32_t ch = b[zk];
            if (ch == 'l' || ch == 's' || ch == 'z') zk++;
         }
      }
      else if (m(zb, zk, zj) == 1 && cvc(zb, zk, zj, zk)) setto(zb, zk, zj, "\01" "e");
   }
}

__forceinline__ __device__ void step1c(char* zb, int32_t& zk, int32_t& zj)
{
   if (ends(zb, zk, zj, "\01" "y") && vowelinstem(zb, zk, zj)) zb[zk] = 'i';
}

__forceinline__ __device__ void step2(char* zb, int32_t& zk, int32_t& zj) 
{ 
    switch (zb[zk-1])
    {
       case 'a': if (ends(zb, zk, zj, "\07" "ational")) { r(zb, zk, zj, "\03" "ate"); break; }
                 if (ends(zb, zk, zj, "\06" "tional")) { r(zb, zk, zj, "\04" "tion"); break; }
                 break;
       case 'c': if (ends(zb, zk, zj, "\04" "enci")) { r(zb, zk, zj, "\04" "ence"); break; }
                 if (ends(zb, zk, zj, "\04" "anci")) { r(zb, zk, zj, "\04" "ance"); break; }
                 break;
       case 'e': if (ends(zb, zk, zj, "\04" "izer")) { r(zb, zk, zj, "\03" "ize"); break; }
                 break;
       case 'l': if (ends(zb, zk, zj, "\03" "bli")) { r(zb, zk, zj, "\03" "ble"); break; } /*-DEPARTURE-*/
                 if (ends(zb, zk, zj, "\04" "alli")) { r(zb, zk, zj, "\02" "al"); break; }
                 if (ends(zb, zk, zj, "\05" "entli")) { r(zb, zk, zj, "\03" "ent"); break; }
                 if (ends(zb, zk, zj, "\03" "eli")) { r(zb, zk, zj, "\01" "e"); break; }
                 if (ends(zb, zk, zj, "\05" "ousli")) { r(zb, zk, zj, "\03" "ous"); break; }
                 break;
       case 'o': if (ends(zb, zk, zj, "\07" "ization")) { r(zb, zk, zj, "\03" "ize"); break; }
                 if (ends(zb, zk, zj, "\05" "ation")) { r(zb, zk, zj, "\03" "ate"); break; }
                 if (ends(zb, zk, zj, "\04" "ator")) { r(zb, zk, zj, "\03" "ate"); break; }
                 break;
       case 's': if (ends(zb, zk, zj, "\05" "alism")) { r(zb, zk, zj, "\02" "al"); break; }
                 if (ends(zb, zk, zj, "\07" "iveness")) { r(zb, zk, zj, "\03" "ive"); break; }
                 if (ends(zb, zk, zj, "\07" "fulness")) { r(zb, zk, zj, "\03" "ful"); break; }
                 if (ends(zb, zk, zj, "\07" "ousness")) { r(zb, zk, zj, "\03" "ous"); break; }
                 break;
       case 't': if (ends(zb, zk, zj, "\05" "aliti")) { r(zb, zk, zj, "\02" "al"); break; }
                 if (ends(zb, zk, zj, "\05" "iviti")) { r(zb, zk, zj, "\03" "ive"); break; }
                 if (ends(zb, zk, zj, "\06" "biliti")) { r(zb, zk, zj, "\03" "ble"); break; }
                 break;
       case 'g': if (ends(zb, zk, zj, "\04" "logi")) { r(zb, zk, zj, "\03" "log"); break; } /*-DEPARTURE-*/


    } 
}

__forceinline__ __device__ void step3(char* zb, int32_t& zk, int32_t& zj) 
{ switch (zb[zk])
    {
       case 'e': if (ends(zb, zk, zj, "\05" "icate")) { r(zb, zk, zj, "\02" "ic"); break; }
                 if (ends(zb, zk, zj, "\05" "ative")) { r(zb, zk, zj, "\00" ""); break; }
                 if (ends(zb, zk, zj, "\05" "alize")) { r(zb, zk, zj, "\02" "al"); break; }
                 break;
       case 'i': if (ends(zb, zk, zj, "\05" "iciti")) { r(zb, zk, zj, "\02" "ic"); break; }
                 break;
       case 'l': if (ends(zb, zk, zj, "\04" "ical")) { r(zb, zk, zj, "\02" "ic"); break; }
                 if (ends(zb, zk, zj, "\03" "ful")) { r(zb, zk, zj, "\00" ""); break; }
                 break;
       case 's': if (ends(zb, zk, zj, "\04" "ness")) { r(zb, zk, zj, "\00" ""); break; }
                 break;
    } 
}

__forceinline__ __device__ void step4(char* zb, int32_t& zk, int32_t& zj)
{  switch (zb[zk-1])
   {  case 'a': if (ends(zb, zk, zj, "\02" "al")) break; return;
      case 'c': if (ends(zb, zk, zj, "\04" "ance")) break;
                if (ends(zb, zk, zj, "\04" "ence")) break; return;
      case 'e': if (ends(zb, zk, zj, "\02" "er")) break; return;
      case 'i': if (ends(zb, zk, zj, "\02" "ic")) break; return;
      case 'l': if (ends(zb, zk, zj, "\04" "able")) break;
                if (ends(zb, zk, zj, "\04" "ible")) break; return;
      case 'n': if (ends(zb, zk, zj, "\03" "ant")) break;
                if (ends(zb, zk, zj, "\05" "ement")) break;
                if (ends(zb, zk, zj, "\04" "ment")) break;
                if (ends(zb, zk, zj, "\03" "ent")) break; return;
      case 'o': if (ends(zb, zk, zj, "\03" "ion") && zj >= 0 && (zb[zj] == 's' || zb[zj] == 't')) break;
                if (ends(zb, zk, zj, "\02" "ou")) break; return;
                /* takes care of -ous */
      case 's': if (ends(zb, zk, zj, "\03" "ism")) break; return;
      case 't': if (ends(zb, zk, zj, "\03" "ate")) break;
                if (ends(zb, zk, zj, "\03" "iti")) break; return;
      case 'u': if (ends(zb, zk, zj, "\03" "ous")) break; return;
      case 'v': if (ends(zb, zk, zj, "\03" "ive")) break; return;
      case 'z': if (ends(zb, zk, zj, "\03" "ize")) break; return;
      default: return;
   }
   if (m(zb, zk, zj) > 1) zk = zj;
}

__forceinline__ __device__ void step5(char* zb, int32_t& zk, int32_t& zj)
{
   char * b = zb;
   zj = zk;
   if (b[zk] == 'e')
   {  int32_t a = m(zb, zk, zj);
      if (a > 1 || a == 1 && !cvc(zb, zk, zj, zk - 1)) zk--;
   }
   if (b[zk] == 'l' && doublec(zb, zk, zj, zk) && m(zb, zk, zj) > 1) zk--;
}

__forceinline__ __device__ int32_t stem_func(char * b, int32_t zk, char* zb)
{

   int32_t zj = 0; 
   if (zk <= 1) return zk; /*-DEPARTURE-*/
   

   step1ab(zb, zk, zj);
   if (zk > 0) {
      step1c(zb, zk, zj); 
      step2(zb, zk, zj); 
      step3(zb, zk, zj); 
      step4(zb, zk, zj); 
      step5(zb, zk, zj);
   }
   return zk;
}

};
#endif // _GPU_PSA_FUNCTION_
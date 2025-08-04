import customAxios from "../utils/customAxios";

type ProfileBlacklist = {
  avata: string;
  full_name: string;
  year_of_birth: number;
  hometown: string;
  id_number: string;
  violation: string;
  data: string[];
};

export const getAllData = async () => {
  try {
    const response = await customAxios.get<ProfileBlacklist>("profileblacklist/");
    return response.data;
  } catch (error) {
    console.log("getProfileBlacklistError: ", error);
  }
};
